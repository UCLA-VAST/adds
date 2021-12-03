/*
 * server_comm.h
 *
 *  Created on: May 29, 2018
 *      Author: redudie
 */

#ifndef SERVER_COMM_H_
#define SERVER_COMM_H_
#define NOT_FOUND 0xffffffff
#define FULL_MASK 0xffffffff
__device__ __forceinline__ void break_pt() {
	asm volatile (
			"brkpt;"
	);
}


__device__ __forceinline__ unsigned load_cg(uint* addr) {
	uint ret_val;
	asm volatile (
			"ld.global.cg.u32 %0, [%1];"
			: "=r" (ret_val) : "l"(addr)
	);
	return ret_val;
}

__device__ __forceinline__ void store_cg(uint* addr, uint val) {
	asm volatile (
			"st.global.cg.f32 [%0],%1;"
			::"l" (addr), "r"(val)
	);
}

__device__ __forceinline__ unsigned find_ms_bit(uint bit_mask) {
	uint ret_val;
	asm volatile (
			"bfind.u32 %0, %1;"
			: "=r" (ret_val) : "r"(bit_mask)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned count_bit(uint bit_mask) {
	uint ret_val;
	asm volatile (
			"popc.b32 %0, %1;"
			: "=r" (ret_val) : "r"(bit_mask)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned extract_bits(uint bit_mask, uint start_pos,
		uint len) {
	uint ret_val;
	asm volatile (
			"bfe.u32 %0, %1, %2, %3;"
			: "=r" (ret_val) : "r"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned set_bits(uint bit_mask, uint val,
		uint start_pos, uint len) {
	uint ret_val;
	asm volatile (
			"bfi.b32 %0, %1, %2, %3, %4;"
			: "=r" (ret_val) :"r" (val), "r"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned long long extract_bits_64(unsigned long long bit_mask, uint start_pos,
		uint len) {
	unsigned long long ret_val;
	asm volatile (
			"bfe.u64 %0, %1, %2, %3;"
			: "=l" (ret_val) : "l"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}


__device__ __forceinline__ unsigned long long set_bits_64(unsigned long long bit_mask, unsigned long long val,
		uint start_pos, uint len) {
	unsigned long long ret_val;
	asm volatile (
			"bfi.b64 %0, %1, %2, %3, %4;"
			: "=l" (ret_val) :"l" (val), "l"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}




__device__ __forceinline__ unsigned find_nth_bit(uint bit_mask, uint base,
		uint offset) {
	uint ret_val;
	asm volatile (
			"fns.b32 %0, %1, %2, %3;"
			: "=r" (ret_val) :"r" (bit_mask), "r"(base), "r" (offset)
	);
	return ret_val;
}

__forceinline__ __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}



__forceinline__ __device__ unsigned get_clock32() {
	unsigned ret;
	asm volatile ("mov.u32 %0, %clock;" : "=r"(ret));
	return ret;
}


__forceinline__ __device__ unsigned thread_id_x() {
	return threadIdx.x;
}

__forceinline__ __device__ unsigned block_id_x() {
	return blockIdx.x;
}

__forceinline__ __device__ unsigned block_dim_x() {
	return blockDim.x;
}

__forceinline__ __device__ unsigned grid_dim_x() {
	return gridDim.x;
}

inline static __device__ __host__ int round_up(int a, int r) {
  return ((a + r - 1) / r) * r;
}

#endif /* SERVER_COMM_H_ */
