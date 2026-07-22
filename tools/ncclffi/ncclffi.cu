// N1 spike extension (nccl-ffi-plan_v1.txt §3.1): our OWN ncclComm_t (bootstrapped
// via mpi4py uid-bcast, NO jax.distributed / NO XLA collective runtime) + an XLA FFI
// handler that enqueues grouped ncclSend/ncclRecv on the XLA compute stream.
// This is the seed of the production "nicam_halo_exchange"; N1 exercises it as a
// single-peer ring under jit / lax.scan / two-independent-exchanges ordering probe.
#include <cstring>

#include <cuda_runtime.h>
#include <nccl.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static ncclComm_t g_comm = nullptr;

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
