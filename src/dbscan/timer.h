#include <mutex>
#include <sys/time.h>
#include <iostream>

using namespace std;

class Timer{
  public:

  double timebase;

  double build_bvh;
  double out_stride;
  double rebuild_bvh;
  double in_stride;
  double find_cores;
  double set_cluster_id;
  double union_cluster_id;
  double copy_cluster_d2h;

  double cuda_find_neighbors;
  double cuda_set_clusters;

  double total;

  Timer() {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
    clear();
  }

  void clear() {
    build_bvh = 0;
    out_stride = 0;
    rebuild_bvh = 0;
    in_stride = 0;
    find_cores = 0;
    set_cluster_id = 0;
    union_cluster_id = 0;
    copy_cluster_d2h = 0;

    cuda_find_neighbors = 0;
    cuda_set_clusters = 0;

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
    
    cout << "[Time] out_stride: " << out_stride / query_num << " ms" << endl;
    cout << "[Time] rebuild_bvh: " << rebuild_bvh / query_num << " ms" << endl;
    cout << "[Time] in_stride: " << in_stride / query_num << " ms" << endl;
    cout << "[Time] find_cores: " << find_cores / query_num << " ms" << endl;
    cout << "[Time] set_cluster_id: " << set_cluster_id / query_num << " ms" << endl;
    cout << "[Time] union_cluster_id: " << union_cluster_id / query_num << " ms" << endl;
    cout << "[Time] copy_cluster_d2h: " << copy_cluster_d2h / query_num << " ms" << endl;
    
    cout << "[Time] cuda_find_neighbors: " << cuda_find_neighbors / query_num << " ms" << endl;
    cout << "[Time] cuda_set_clusters: " << cuda_set_clusters / query_num << " ms" << endl;
    
    cout << "[Time] total: " << total / query_num << " ms" << endl;
    // cout << "[Time] total: " << copy_queries_h2d + search_neighbors + copy_results_d2h << " ms" << endl;

    cout << "##############################" << endl;
    cout << endl;
  }
};
