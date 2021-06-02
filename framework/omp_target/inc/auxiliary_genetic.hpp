

#ifndef AUXILIARY_GENETIC_H  
#define AUXILIARY_GENETIC_H  

inline uint32_t gpu_rand(
                uint32_t* prng_states,
                int blockIdx, int threadIdx
);

inline float gpu_randf(
                uint32_t* prng_states,
                int blockIdx, int threadIdx
);

inline void map_angle(
		float& angle
);

#endif
