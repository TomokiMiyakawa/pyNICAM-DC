// N1 spike extension (nccl-ffi-plan_v1.txt §3.1): our OWN ncclComm_t (bootstrapped
// via mpi4py uid-bcast, NO jax.distributed / NO XLA collective runtime) + an XLA FFI
// handler that enqueues grouped ncclSend/ncclRecv on the XLA compute stream.
// This is the seed of the production "nicam_halo_exchange"; N1 exercises it as a
// single-peer ring under jit / lax.scan / two-independent-exchanges ordering probe.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static ncclComm_t g_comm = nullptr;
static int g_rank = -1;

extern "C" int ncclffi_uid_size() { return (int)sizeof(ncclUniqueId); }

extern "C" int ncclffi_get_uid(char* out) {
  ncclUniqueId id;
  ncclResult_t r = ncclGetUniqueId(&id);
  if (r != ncclSuccess) return (int)r;
  std::memcpy(out, &id, sizeof(id));
  return 0;
}

// Blocking-collective across ranks: every rank must call with the same uid.
extern "C" int ncclffi_init(int rank, int nranks, const char* uid_bytes, int device) {
  cudaError_t ce = cudaSetDevice(device);
  if (ce != cudaSuccess) return 1000 + (int)ce;
  ncclUniqueId id;
  std::memcpy(&id, uid_bytes, sizeof(id));
  g_rank = rank;
  return (int)ncclCommInitRank(&g_comm, nranks, id, rank);
}

extern "C" int ncclffi_finalize() {
  if (g_comm) { ncclCommDestroy(g_comm); g_comm = nullptr; }
  return 0;
}

// One point-to-point exchange: send the whole buffer to send_peer, receive the
// same count from recv_peer. Grouped -> atomic NCCL launch, no host sync.
static ffi::Error PeerExchangeImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> x,
                                   ffi::ResultBuffer<ffi::F32> y,
                                   int64_t send_peer, int64_t recv_peer) {
  if (!g_comm) return ffi::Error::Internal("ncclffi: not initialized");
  size_t n = x.element_count();
  ncclGroupStart();
  ncclSend(x.typed_data(), n, ncclFloat32, (int)send_peer, g_comm, stream);
  ncclRecv(y->typed_data(), n, ncclFloat32, (int)recv_peer, g_comm, stream);
  ncclResult_t r = ncclGroupEnd();
  if (r != ncclSuccess) return ffi::Error::Internal(ncclGetErrorString(r));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PeerExchange, PeerExchangeImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()   // x
                                  .Ret<ffi::Buffer<ffi::F32>>()   // y
                                  .Attr<int64_t>("send_peer")
                                  .Attr<int64_t>("recv_peer"));

// ---------------------------------------------------------------------------
// N2 production handler: "nicam_halo_exchange" (nccl-ffi-plan_v1.txt §3.1b).
// Per-partner offset/count tables are registered HOST-SIDE at setup
// (ncclffi_set_plan, from mod_comm's a2a_send/a2a_recv) and referenced by a
// scalar plan_id attr -- no attr-array machinery, no device-side index reads.
// Offsets/counts are in ELEMENTS of the buffer dtype (fp32 or fp64).
// ---------------------------------------------------------------------------
struct HaloPlan {
  std::vector<int> peers;
  std::vector<long long> send_off, send_cnt, recv_off, recv_cnt;
};
static std::map<int, HaloPlan> g_plans;

extern "C" int ncclffi_set_plan(int plan_id, int n_partners, const long long* peers,
                                const long long* send_off, const long long* send_cnt,
                                const long long* recv_off, const long long* recv_cnt) {
  HaloPlan p;
  for (int i = 0; i < n_partners; ++i) p.peers.push_back((int)peers[i]);
  p.send_off.assign(send_off, send_off + n_partners);
  p.send_cnt.assign(send_cnt, send_cnt + n_partners);
  p.recv_off.assign(recv_off, recv_off + n_partners);
  p.recv_cnt.assign(recv_cnt, recv_cnt + n_partners);
  g_plans[plan_id] = p;
  return 0;
}

// tok_in/tok_out: 1-float ordering token threaded PHYSICALLY through every call.
// An optimization_barrier tie is HLO-only -- it emits no thunk, so XLA:GPU's
// buffer-level thunk scheduler dropped it and reordered two independent
// exchanges differently on different ranks (N2h: tracer pair swapped on ranks
// 1/3 => tag-less NCCL cross-matched payloads). Making the token a real custom-
// call operand/result creates a hard buffer def-use edge the scheduler must keep.
static ffi::Error HaloExchangeImpl(cudaStream_t stream, ffi::AnyBuffer x,
                                   ffi::Buffer<ffi::F32> tok_in,
                                   ffi::Result<ffi::AnyBuffer> y,
                                   ffi::Result<ffi::Buffer<ffi::F32>> tok_out,
                                   int64_t plan_id) {
  if (!g_comm) return ffi::Error::Internal("ncclffi: not initialized");
  // Diagnostic (PYNICAM_NCCLFFI_SYNC=1): full device sync before the sends --
  // if a cross-stream pack/lifetime race feeds NCCL stale operand bytes, this
  // closes it and the model A/B goes to 0.0.
  static const int g_sync = [] {
    const char* e = getenv("PYNICAM_NCCLFFI_SYNC");
    return (e && e[0] != '0') ? 1 : 0;
  }();
  if (g_sync) cudaDeviceSynchronize();
  auto it = g_plans.find((int)plan_id);
  if (it == g_plans.end()) return ffi::Error::Internal("ncclffi: unknown plan_id");
  const HaloPlan& p = it->second;
  ncclDataType_t nt;
  size_t esz;
  switch (x.element_type()) {
    case ffi::DataType::F32: nt = ncclFloat32; esz = 4; break;
    case ffi::DataType::F64: nt = ncclFloat64; esz = 8; break;
    default: return ffi::Error::Internal("ncclffi: unsupported dtype");
  }
  const char* xb = static_cast<const char*>(x.untyped_data());
  char* yb = static_cast<char*>(y->untyped_data());
  // thread the ordering token: tok_out <- tok_in (value irrelevant, the buffer
  // dependency is everything)
  cudaMemcpyAsync(tok_out->untyped_data(), tok_in.untyped_data(), sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  ncclGroupStart();
  for (size_t i = 0; i < p.peers.size(); ++i) {
    if (p.send_cnt[i] > 0)
      ncclSend(xb + (size_t)p.send_off[i] * esz, (size_t)p.send_cnt[i], nt,
               p.peers[i], g_comm, stream);
    if (p.recv_cnt[i] > 0)
      ncclRecv(yb + (size_t)p.recv_off[i] * esz, (size_t)p.recv_cnt[i], nt,
               p.peers[i], g_comm, stream);
  }
  ncclResult_t r = ncclGroupEnd();
  if (r != ncclSuccess) return ffi::Error::Internal(ncclGetErrorString(r));

  // Diagnostic (PYNICAM_NCCLFFI_CKSUM=1): per-call, per-peer XOR-fold checksums of
  // the SENT and RECEIVED byte segments. Cross-matched offline: sender(r->q).send
  // must equal receiver(q).recv-from-r at the same call#. Exact (no fp), catches
  // any in-flight byte change at the precise call/pair.
  static const int g_cksum = [] {
    const char* e = getenv("PYNICAM_NCCLFFI_CKSUM");
    return (e && e[0] != '0') ? 1 : 0;
  }();
  if (g_cksum) {
    static int call_no = 0;
    ++call_no;
    cudaStreamSynchronize(stream);
    size_t xb_bytes = x.size_bytes(), yb_bytes = y->size_bytes();
    static std::vector<unsigned char> hx, hy;
    hx.resize(xb_bytes); hy.resize(yb_bytes);
    cudaMemcpy(hx.data(), xb, xb_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), yb, yb_bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < p.peers.size(); ++i) {
      auto fold = [](const unsigned char* b, size_t n) {
        unsigned long long acc = 0, w = 0;
        for (size_t k = 0; k < n; ++k) {
          w = (w << 8) | b[k];
          if ((k & 7) == 7) { acc ^= w; w = 0; }
        }
        if (n & 7) acc ^= w;
        return acc;
      };
      unsigned long long cs = p.send_cnt[i] > 0
          ? fold(hx.data() + (size_t)p.send_off[i] * esz, (size_t)p.send_cnt[i] * esz) : 0;
      unsigned long long cr = p.recv_cnt[i] > 0
          ? fold(hy.data() + (size_t)p.recv_off[i] * esz, (size_t)p.recv_cnt[i] * esz) : 0;
      fprintf(stderr, "NCCLFFI_CKSUM call=%d plan=%d rank=%d peer=%d send=%016llx recv=%016llx\n",
              call_no, (int)plan_id, g_rank, p.peers[i], cs, cr);
    }
    fflush(stderr);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(HaloExchange, HaloExchangeImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()          // packed send buffer
                                  .Arg<ffi::Buffer<ffi::F32>>()   // ordering token in
                                  .Ret<ffi::AnyBuffer>()          // packed recv buffer
                                  .Ret<ffi::Buffer<ffi::F32>>()   // ordering token out
                                  .Attr<int64_t>("plan_id"));
