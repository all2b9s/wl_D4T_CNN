#pragma once
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#if defined(__linux__)
#include <sched.h>
#endif
namespace d4cnn {
inline std::vector<int> allowed_cores(){
 std::vector<int> cores;
#if defined(__linux__)
 cpu_set_t mask;CPU_ZERO(&mask);
 if(sched_getaffinity(0,sizeof(mask),&mask)==0)
  for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&mask))cores.push_back(cpu);
#endif
 return cores;
}
// One executor per calling thread. Concurrent callers have independent jobs;
// workers survive batches and retain CNN scratch and AnaCal ImageQLease caches.
class Executor {
 std::mutex mutex;
 std::condition_variable ready,done;
 std::vector<std::thread> workers;
 std::vector<int> cores;
 std::function<void(int)> task;
 std::atomic<int> next{0};
 std::atomic<bool> failed{false};
 std::exception_ptr error;
 int count=0,remaining=0;
 unsigned long generation=0;
 bool stopping=false;
 void work(int index){
#if defined(__linux__)
  if(!cores.empty()){cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cores[index%cores.size()],&mask);
   // Failure leaves the inherited allocation mask in place.
   sched_setaffinity(0,sizeof(mask),&mask);}
#endif
  unsigned long seen=0;
  for(;;){
   {std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock,[&]{return stopping||generation!=seen;});
    if(stopping)return;seen=generation;
   }
   try{while(!failed.load(std::memory_order_relaxed)){
    int i=next.fetch_add(1,std::memory_order_relaxed);if(i>=count)break;task(i);
   }}catch(...){std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();failed=true;}
   {std::lock_guard<std::mutex> lock(mutex);if(--remaining==0)done.notify_one();}
  }
 }
 void stop(){
  {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
  ready.notify_all();for(auto& worker:workers)worker.join();
 }
public:
 explicit Executor(int threads,std::vector<int> affinity):cores(std::move(affinity)){
  try{for(int i=0;i<threads;++i)workers.emplace_back([this,i]{work(i);});}
  catch(...){stop();throw;}
 }
 ~Executor(){stop();}
 int size()const{return int(workers.size());}
 bool matches(int threads,const std::vector<int>& affinity)const{return size()==threads&&cores==affinity;}
 void run(int n,std::function<void(int)> fn){
  std::unique_lock<std::mutex> lock(mutex);
  task=std::move(fn);count=n;next=0;failed=false;error=nullptr;
  remaining=size();++generation;ready.notify_all();
  done.wait(lock,[&]{return remaining==0;});
  task=nullptr;if(error)std::rethrow_exception(error);
 }
};
inline Executor& executor(int threads){
 thread_local std::unique_ptr<Executor> pool;
 auto affinity=allowed_cores();
 if(!pool||!pool->matches(threads,affinity))pool=std::make_unique<Executor>(threads,std::move(affinity));
 return *pool;
}
template<class F> void parallel_for(int n,int threads,F fn){
 if(n==0)return;
 if(threads==1){for(int i=0;i<n;++i)fn(i);return;}
 executor(threads).run(n,std::move(fn));
}
}
