#include <mutex>
#include <sys/time.h>
#include <iostream>

using namespace std;

class Timer{
  public:

  double timebase;

  double copy_queries_h2d;
  double search_neighbors;
  double cuda_refine;
  double copy_results_d2h;

  double total;

  Timer() {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
    clear();
  }

  void clear() {
    copy_queries_h2d = 0;
    search_neighbors = 0;
    cuda_refine      = 0;
    copy_results_d2h = 0;
    total = 0;
  }
  
  // void commonGetStartTime(int timeId) {
  //   struct timeval t1;                           
  //   gettimeofday(&t1, NULL);
  //   lock_guard<mutex> lock(timeMutex[timeId]);
  //   time[timeId] -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  // }

  // void commonGetEndTime(int timeId) {
  //   struct timeval t1;                           
  //   gettimeofday(&t1, NULL);
  //   lock_guard<mutex> lock(timeMutex[timeId]);
  //   time[timeId] += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  // }

  void startTimer(double *t) {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    *t -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }

  void stopTimer(double *t) {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    *t += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }


  // void average(int n) {
  //   copy_new_points_h2d /= n;
  //   copy_filtered_points_h2d /= n;
  //   copy_outlier_d2h /= n;
  //   prepare_cell /= n;
  //   build_bvh /= n;
  //   detect_outlier /= n;
  //   total /= n;
  // }

  void showTime(int query_num) {
    cout << endl;
    cout << "###########   Time  ##########" << endl;
    
    cout << "[Time] copy queries h2d: " << copy_queries_h2d / query_num << " ms" << endl;
    cout << "[Time] search neighbors: " << search_neighbors / query_num << " ms" << endl;
    cout << "[Time] cuda refine: " << cuda_refine / query_num << " ms" << endl;
    cout << "[Time] copy results d2h: " << copy_results_d2h / query_num << " ms" << endl;
    // cout << "[Time] total: " << copy_queries_h2d + search_neighbors + copy_results_d2h << " ms" << endl;

    cout << "##############################" << endl;
    cout << endl;
  }
};
