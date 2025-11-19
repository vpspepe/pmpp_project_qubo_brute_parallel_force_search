#pragma once
#include "cuda_util.h"
#include <string>
#include <iostream>
#include <vector>


class CudaTimer
{
public:
	CudaTimer(CudaTimer& t) = delete;
	CudaTimer() {}
	CudaTimer(std::string name, bool blocking = false, cudaStream_t stream = nullptr)
		: name(name)
		, stream(stream)
	{
		if (blocking) {
			CUDA_CALL(cudaEventCreateWithFlags(&begin, cudaEventBlockingSync));
			CUDA_CALL(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
		}
		else {
			CUDA_CALL(cudaEventCreate(&begin));
			CUDA_CALL(cudaEventCreate(&end));
		}
		
		start();
	}
	~CudaTimer() { if (!stopped) stop(); }
	inline void start() {
		stopped = false;
		CUDA_CALL(cudaEventRecord(begin, stream));
	}
	inline void stop() {
		if (!stopped) {
			CUDA_CALL(cudaEventRecord(end, stream));
		}
		stopped = true;
	}
	inline float wait_for_time() {
		float time;
		CUDA_CALL(cudaEventSynchronize(end));
		bool success = CUDA_CALL(cudaEventElapsedTime(&time, begin, end));
		if (!success)
			time = -1.f;
		
		std::cout<< "Time for " << name << ": " << time << " ms\n";
		return time;
	}
	inline void start_new(std::string new_name) {
		stop();
		this->name = new_name;
		start();
	}
	inline std::string get_name() {
		return name;
	}
private:
	std::string name;
	cudaStream_t stream = nullptr;
	cudaEvent_t begin = nullptr, end = nullptr;
	
	bool stopped = true, blocking = false;
};

class CudaTimerManager {
public:
	void add(CudaTimer& timer) {
		timers.push_back(&timer);
		names.push_back(timer.get_name());
	}

	void wait_and_print_all() {
		for (auto& t : timers) {
			results.push_back(t->wait_for_time());
		}
	}
	std::vector<std::string> names;
	std::vector<float> results;
private:
	std::vector<CudaTimer*> timers;
};