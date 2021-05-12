#
# miniAD Makefile
# ------------------------------------------------------
# Note that environment variables must be defined
# before compiling
# DEVICE?
# if DEVICE=CPU: CPU_INCLUDE_PATH?, CPU_LIBRARY_PATH?
# if DEVICE=GPU: GPU_INCLUDE_PATH?, GPU_LIBRARY_PATH?
#
# ------------------------------------------------------

# DEVICE 
# Possible values for DEVICE: CPU, GPU
# API
# Possible values for API: CUDA, HIP, KOKKOS
# VENDOR
# Possible values for CARD: NVIDIA, AMD


#ifeq ($(DEVICE), $(filter $(DEVICE),GPU CUDA))
ifeq ($(API), CUDA)
TEST_CUDA := $(shell ./test_cuda.sh nvcc "$(GPU_INCLUDE_PATH)" "$(GPU_LIBRARY_PATH)")
# if user specifies API=CUDA it will be used (wether the test succeeds or not)
# if user specifies DEVICE=GPU the test result determines wether CUDA will be used or not
ifeq ($(API)$(TEST_CUDA),GPUyes)
override API:=CUDA
endif
endif

ifeq ($(API),CUDA)
 override DEVICE:=GPU
 export
 include Makefile.Cuda
else
ifeq ($(API),HIP)
  override DEVICE:=GPU
  export
  ifeq ($(CARD), NVIDIA)
   include Makefile.Hip.Nvidia
  else
  ifeq ($(CARD), AMD)
   include Makefile.Hip.Amd
  endif
  endif
else
ifeq ($(API), KOKKOS)
  ifeq ($(DEVICE), SERIAL)
    override DEVICE:=SERIAL
    export
  else
    override DEVICE:=GPU
    export
    ifeq ($(CARD), NVIDIA)
      override CARD:=NVIDIA
      export
    else
    ifeq ($(CARD), AMD)
      override CARD:=AMD
      export
    endif
    endif
  endif
  include Makefile.Kokkos
endif
endif
endif

