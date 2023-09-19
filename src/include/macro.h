#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <bitset>
#include <vector>
#include <thrust/complex.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

namespace ModalSound
{
#define MLOG_ERROR_COLOR "\033[1;31m"
#define MLOG_WARNING_COLOR "\033[1;33m"
#define MLOG_INFO_COLOR "\033[1;32m"
#define MLOG_DEBUG_COLOR "\033[1;34m"
#define MLOG_RESET_COLOR "\033[0m"
#define MLOG_BOLD_COLOR "\033[1m"

#define MLOG_FILE std::cout
#define MLOG(x) MLOG_FILE << x << std::endl;
#define MLOG_ERROR(x) MLOG_FILE << MLOG_ERROR_COLOR << x << MLOG_RESET_COLOR << std::endl;
#define MLOG_WARNING(x) MLOG_FILE << MLOG_WARNING_COLOR << x << MLOG_RESET_COLOR << std::endl;
#define MLOG_INFO(x) MLOG_FILE << MLOG_INFO_COLOR << x << MLOG_RESET_COLOR << std::endl;
#define MLOG_DEBUG(x) MLOG_FILE << MLOG_DEBUG_COLOR << x << MLOG_RESET_COLOR << std::endl;
#define MLOG_LINE(x) MLOG_FILE << MLOG_BOLD_COLOR << "---------------" << x << "---------------" << MLOG_RESET_COLOR << std::endl;
#define MLOG_INLINE(x) MLOG_FILE << x;

#define TICK_PRECISION 3
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TICK_INLINE(x) auto bench_inline_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                                                                                                                 \
	std::streamsize ss_##x = std::cout.precision();                                                                                                             \
	std::cout.precision(TICK_PRECISION);                                                                                                                        \
	MLOG_FILE << #x ": " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count() << "s" << std::endl; \
	std::cout.precision(ss_##x);
#define TOCK_INLINE(x)                                                                                                                                   \
	std::streamsize ss_##x = std::cout.precision();                                                                                                      \
	std::cout.precision(TICK_PRECISION);                                                                                                                 \
	MLOG_FILE << #x ": " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_inline_##x).count() << "s" \
			 << "  \t";                                                                                                                                  \
	std::cout.precision(ss_##x);

#define PI 3.14159265359
#define CGPU_FUNC __host__ __device__
#define GPU_FUNC __device__
#define CPU_FUNC __host__
#define RAND_F (float)rand() / (float)RAND_MAX
#define RAND_I(min, max) (min + rand() % (max - min + 1))
#define BDF2(x) 0.5 * ((x) * (x)-4 * (x) + 3)

	typedef float real_t;
	const real_t EPS = 1e-8f;
	typedef thrust::complex<real_t> cpx;

	enum cpx_phase
	{
		CPX_REAL,
		CPX_IMAG,
		CPX_ABS,
	};

	enum PlaneType
	{
		XY,
		XZ,
		YZ,
	};

	typedef std::vector<real_t> RealVec;
	typedef std::vector<cpx> ComplexVec;

	static inline void checkCudaError(const char *msg)
	{
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			// printf( "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);
			throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
		}
	}

#define atomicAddCpx(dst, value)                         \
	{                                                    \
		atomicAdd((float *)(dst), (value).real());       \
		atomicAdd(((float *)(dst)) + 1, (value).imag()); \
	}
#define atomicAddCpxBlock(dst, value)                          \
	{                                                          \
		atomicAdd_block((float *)(dst), (value).real());       \
		atomicAdd_block(((float *)(dst)) + 1, (value).imag()); \
	}

#ifdef NDEBUG
#define cuSafeCall(X) X
#else
#define cuSafeCall(X) \
	X;                \
	checkCudaError(#X);
#endif

/**
 * @brief Macro to check cuda errors
 *
 */
#ifdef NDEBUG
#define cuSynchronize() \
	{                   \
	}
#else
#define cuSynchronize()                                                                                        \
	{                                                                                                          \
		char str[200];                                                                                         \
		cudaDeviceSynchronize();                                                                               \
		cudaError_t err = cudaGetLastError();                                                                  \
		if (err != cudaSuccess)                                                                                \
		{                                                                                                      \
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
			throw std::runtime_error(std::string(str));                                                        \
		}                                                                                                      \
	}
#endif

	static uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	static uint cudaGridSize(uint totalSize, uint blockSize)
	{
		int dim = iDivUp(totalSize, blockSize);
		return dim == 0 ? 1 : dim;
	}

#define CUDA_BLOCK_SIZE 64
	/**
	 * @brief Macro definition for execuation of cuda kernels, note that at lease one block will be executed.
	 *
	 * size: indicate how many threads are required in total.
	 * Func: kernel function
	 */

#define cuExecute(size, Func, ...)                              \
	{                                                           \
		uint pDims = cudaGridSize((uint)size, CUDA_BLOCK_SIZE); \
		Func<<<pDims, CUDA_BLOCK_SIZE>>>(                       \
			__VA_ARGS__);                                       \
		cuSynchronize();                                        \
	}

#define cuExecuteBlock(size1, size2, Func, ...) \
	{                                           \
		Func<<<size1, size2>>>(                 \
			__VA_ARGS__);                       \
		cuSynchronize();                        \
	}

#define cuExecuteBlockDym(size1, size2, size3, Func, ...) \
	{                                                     \
		Func<<<size1, size2, size3>>>(                    \
			__VA_ARGS__);                                 \
		cuSynchronize();                                  \
	}

#define SHOW(x) std::cout << #x << ":\n" \
						  << x << std::endl;

	static std::ostream &operator<<(std::ostream &o, float3 const &f)
	{
		return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
	}
	static std::ostream &operator<<(std::ostream &o, uint3 const &f)
	{
		return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
	}
	static std::ostream &operator<<(std::ostream &o, int3 const &f)
	{
		return o << "(" << f.x << ", " << f.y << ", " << f.z << ")";
	}
	static std::ostream &operator<<(std::ostream &o, int4 const &a)
	{
		return o << "(" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << ")";
	}

	class Range
	{
	public:
		uint start;
		uint end;
		Range(uint start, uint end) : start(start), end(end) {}
		Range() : start(0), end(0) {}
		inline int length()
		{
			return end - start;
		}
		friend std::ostream &operator<<(std::ostream &os, const Range &r)
		{
			os << "[" << r.start << "," << r.end << ")";
			return os;
		}
	};

	class BBox
	{
	public:
		float3 min;
		float3 max;
		float width;
		BBox() {}
		BBox(float3 _min, float3 _max)
		{
			min = _min;
			max = _max;
		}
		friend std::ostream &operator<<(std::ostream &os, const BBox &b)
		{
			os << "[" << b.min << "," << b.max << "]";
			return os;
		}
	};
}