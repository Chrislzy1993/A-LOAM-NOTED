// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
 * @file laserOdometry.cpp
 * @brief：根据点云的边缘点和平面点特征进行scan-scan的匹配
 * @input ：cloud, cloud_sharp, cloud_less_sharp, cloud_flat, cloud_less_flatss
 * @output：cloud, odom_path, cloud_corner_last, cloud_surf_last
 */
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen3/Eigen/Dense>
#include <queue>
#include <mutex>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0

int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

bool systemInited = false;
int skipFrameNum = 5;

// 保证接收到的点云是同一时刻的
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

// 对上一帧的点云构建kdtree, 用于快速查找邻近点
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// sub点云
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

// pub点云
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// 里程计的当前位姿转换到世界坐标系下
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// 点云特征匹配时的优化变量
double para_q[4] = { 0, 0, 0, 1 };
double para_t[3] = { 0, 0, 0 };

// 优化变量para_q和para_t的映射：表示的是两个world坐标系下的位姿P之间的增量，例如△P = P0.inverse() * P1
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// 队列
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;  // 互斥锁

// 将点云点插值变换到帧首
void TransformToStart(PointType const* const pi, PointType* const po)
{
  // interpolation ratio
  double s;
  if (DISTORTION)
    s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
  else
    s = 1.0;
  // s = 1;
  Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
  Eigen::Vector3d t_point_last = s * t_last_curr;
  Eigen::Vector3d point(pi->x, pi->y, pi->z);
  Eigen::Vector3d un_point = q_point_last * point + t_point_last;

  po->x = un_point.x();
  po->y = un_point.y();
  po->z = un_point.z();
  po->intensity = pi->intensity;
}

// 将点云点插值变换到帧尾，也就是去除运动畸变
void TransformToEnd(PointType const* const pi, PointType* const po)
{
  // undistort point first
  pcl::PointXYZI un_point_tmp;
  TransformToStart(pi, &un_point_tmp);

  Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
  Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

  po->x = point_end.x();
  po->y = point_end.y();
  po->z = point_end.z();

  // Remove distortion time info
  po->intensity = int(pi->intensity);
}

// 接收点云数据，接收后加入到队列中
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharp2)
{
  mBuf.lock();
  cornerSharpBuf.push(cornerPointsSharp2);
  mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharp2)
{
  mBuf.lock();
  cornerLessSharpBuf.push(cornerPointsLessSharp2);
  mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlat2)
{
  mBuf.lock();
  surfFlatBuf.push(surfPointsFlat2);
  mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlat2)
{
  mBuf.lock();
  surfLessFlatBuf.push(surfPointsLessFlat2);
  mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  mBuf.lock();
  fullPointsBuf.push(laserCloudFullRes2);
  mBuf.unlock();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserOdometry");
  ros::NodeHandle nh;

  nh.param<int>("mapping_skip_frame", skipFrameNum, 2);
  printf("Mapping %d Hz \n", 10 / skipFrameNum);

  // 订阅和发布节点消息
  ros::Subscriber subCornerPointsSharp =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
  ros::Subscriber subCornerPointsLessSharp =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
  ros::Subscriber subSurfPointsFlat =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
  ros::Subscriber subSurfPointsLessFlat =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
  ros::Subscriber subLaserCloudFullRes =
      nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

  ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
  ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
  ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
  ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

  nav_msgs::Path laserPath; // odometry路径
  
  /**
   * @brief:雷达里程计主要框架 
   * @brief:1. 接收特征点云消息，并存入队列中，并同步消息
   * @brief:2. 如果是第一帧，将LessSharp保存为CornerLast，LessFlat保存为SurfLast，同时建立kdtree点云
   * @brief:3. 如果不是第一帧，建立scan-scan的匹配，根据最邻近寻找特征点在上一帧的correspondence-边缘线和平面
   * @brief:4. 建立点线距离和点面距离误差方程，使用ceres优化求解，通过求解的雷达坐标系下的帧间位姿计算出世界坐标系下的当前位姿
   * @brief:5. 将当前帧的LessSharp保存为CornerLast，LessFlat保存为SurfLast，同时建立kdtree点云
   */
  int frameCount = 0;
  ros::Rate rate(100); // 保证odometry处理速度在10Hz左右  
  while (ros::ok())
  {
    ros::spinOnce();
    // 保证各个队列都接收到了点云
    if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() && !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
        !fullPointsBuf.empty())
    {
      timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
      timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
      timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
      timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
      timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

      // 保证是同一时刻点云的消息
      if (timeCornerPointsSharp != timeLaserCloudFullRes || timeCornerPointsLessSharp != timeLaserCloudFullRes ||
          timeSurfPointsFlat != timeLaserCloudFullRes || timeSurfPointsLessFlat != timeLaserCloudFullRes)
      {
        printf("unsync messeage!");
        ROS_BREAK();
      }
      
      // 取队列中的消息
      mBuf.lock();
      cornerPointsSharp->clear();
      pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
      cornerSharpBuf.pop();

      cornerPointsLessSharp->clear();
      pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
      cornerLessSharpBuf.pop();

      surfPointsFlat->clear();
      pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
      surfFlatBuf.pop();

      surfPointsLessFlat->clear();
      pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
      surfLessFlatBuf.pop();

      laserCloudFullRes->clear();
      pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
      fullPointsBuf.pop();
      mBuf.unlock();

      TicToc t_whole;
      if (!systemInited) // 第一帧数据不做处理
      {
        systemInited = true;
        std::cout << "Initialization finished \n";
      }
      else
      {
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();

        TicToc t_opt;
        for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)  // 点到线以及点到面的ICP，迭代2次
        {
          corner_correspondence = 0;
          plane_correspondence = 0;
          
          // 定义ceres优化器
          // ceres::LossFunction *loss_function = NULL;
          ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1); // 损失核函数
          ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
          ceres::Problem::Options problem_options;
          ceres::Problem problem(problem_options);
          problem.AddParameterBlock(para_q, 4, q_parameterization);
          problem.AddParameterBlock(para_t, 3);

          pcl::PointXYZI pointSel;  // 投影点
          std::vector<int> pointSearchInd; // K近邻搜索点
          std::vector<float> pointSearchSqDis; // 搜索点平方距离

          TicToc t_data;
          // 基于最近邻原理建立边缘点的关联-边缘线AB，建立点线距离误差方程
          // 1. 将当前帧的点投影到上一帧o
          // 2. 计算投影点o的最近点A
          // 3. 计算投影点o的次近邻点B且与点A在相邻SCAN上
          // 4. 建立当前帧边缘点到上一帧边缘线AB的点线距离误差方程，使用ceres优化求解
          for (int i = 0; i < cornerPointsSharpNum; ++i)
          {
            // 1.在雷达坐标系下将当前帧的点投影到上一帧
            TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
            // 2. 计算投影点的最近点A  
            kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis); 
            int closestPointInd = -1, minPointInd2 = -1;
            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)  // 如果最近邻的corner特征点之间距离平方小于阈值，则最近邻点A有效
            {
              closestPointInd = pointSearchInd[0];
              int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);
              double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
              // 3. 计算投影点的次近邻点B且与点A在相邻SCAN上
              // search in the direction of increasing scan line ???
              for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)  
              {
                // if in the same scan line, continue
                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)  
                  continue;

                // if not in nearby scans, end the loop 
                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                  break;

                double pointSqDis =
                    (laserCloudCornerLast->points[j].x - pointSel.x) *
                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                    (laserCloudCornerLast->points[j].z - pointSel.z) * (laserCloudCornerLast->points[j].z - pointSel.z);
                // 第二个最近邻点有效，更新点B
                if (pointSqDis < minPointSqDis2)  
                {
                  minPointSqDis2 = pointSqDis;
                  minPointInd2 = j;
                }
              }

              // 3. 计算投影点的次近邻点B且与点A在相邻SCAN上
              // search in the direction of decreasing scan line
              for (int j = closestPointInd - 1; j >= 0; --j)
              {
                // if in the same scan line, continue
                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                  continue;

                // if not in nearby scans, end the loop
                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                  break;

                double pointSqDis =
                    (laserCloudCornerLast->points[j].x - pointSel.x) *
                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                    (laserCloudCornerLast->points[j].z - pointSel.z) * (laserCloudCornerLast->points[j].z - pointSel.z);
                // 第二个最近邻点有效，更新点B
                if (pointSqDis < minPointSqDis2)  
                {
                  minPointSqDis2 = pointSqDis;
                  minPointInd2 = j;
                }
              }
            }
            if (minPointInd2 >= 0)  
            {                       
              Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x, cornerPointsSharp->points[i].y,
                                         cornerPointsSharp->points[i].z);
              Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                           laserCloudCornerLast->points[closestPointInd].y,
                                           laserCloudCornerLast->points[closestPointInd].z);
              Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                           laserCloudCornerLast->points[minPointInd2].y,
                                           laserCloudCornerLast->points[minPointInd2].z);
              
              // 运动补偿系数，kitti数据集的点云已经被补偿过，所以s = 1.0
              double s;  
              if (DISTORTION)
                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) /
                    SCAN_PERIOD;
              else
                s = 1.0;
              
              // 用当前帧点，上一帧对应点A和B构造点到线的距离的残差项，残差项为点到直线的距离方程，具体见lidarFactor.cpp
              ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
              problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
              corner_correspondence++;
            }
          }
          // 基于最近邻原理建立平面点的关联-平面ABC，建立点面距离误差方程
          // 1. 将当前帧的点投影到上一帧得到点o
          // 2. 计算投影点o的最近点A
          // 3. 计算投影点o的次近邻点B和C且和A不在同一个SCAN上
          // 4. 建立当前帧边缘点到上一帧的对应平面ABC的点面距离误差方程，使用ceres优化求解
          for (int i = 0; i < surfPointsFlatNum; ++i)
          {
            // 1. 将当前帧的点投影到上一帧得到点o
            TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
            // 2. 计算投影点o的最近点A
            kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
            int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
            if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)  // 找到的最近邻点A有效
            {
              closestPointInd = pointSearchInd[0];

              // get closest point's scan ID
              int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
              double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

              // search in the direction of increasing scan line
              for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
              {
                // if not in nearby scans, end the loop
                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                  break;

                double pointSqDis =
                    (laserCloudSurfLast->points[j].x - pointSel.x) * (laserCloudSurfLast->points[j].x - pointSel.x) +
                    (laserCloudSurfLast->points[j].y - pointSel.y) * (laserCloudSurfLast->points[j].y - pointSel.y) +
                    (laserCloudSurfLast->points[j].z - pointSel.z) * (laserCloudSurfLast->points[j].z - pointSel.z);

                // if in the same or lower scan line
                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                {
                  minPointSqDis2 = pointSqDis; 
                  minPointInd2 = j;
                }
                // if in the higher scan line
                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                {
                  minPointSqDis3 = pointSqDis;  
                  minPointInd3 = j;
                }
              }

              // search in the direction of decreasing scan line
              for (int j = closestPointInd - 1; j >= 0; --j)
              {
                // if not in nearby scans, end the loop
                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                  break;

                double pointSqDis =
                    (laserCloudSurfLast->points[j].x - pointSel.x) * (laserCloudSurfLast->points[j].x - pointSel.x) +
                    (laserCloudSurfLast->points[j].y - pointSel.y) * (laserCloudSurfLast->points[j].y - pointSel.y) +
                    (laserCloudSurfLast->points[j].z - pointSel.z) * (laserCloudSurfLast->points[j].z - pointSel.z);

                // if in the same or higher scan line
                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                {
                  minPointSqDis2 = pointSqDis;
                  minPointInd2 = j;
                }
                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID &&
                         pointSqDis < minPointSqDis3)
                {
                  // find nearer point
                  minPointSqDis3 = pointSqDis;
                  minPointInd3 = j;
                }
              }

              if (minPointInd2 >= 0 && minPointInd3 >= 0)  // 如果三个最近邻点都有效
              {
                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x, surfPointsFlat->points[i].y,
                                           surfPointsFlat->points[i].z);
                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                             laserCloudSurfLast->points[closestPointInd].y,
                                             laserCloudSurfLast->points[closestPointInd].z);
                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                             laserCloudSurfLast->points[minPointInd2].y,
                                             laserCloudSurfLast->points[minPointInd2].z);
                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                             laserCloudSurfLast->points[minPointInd3].y,
                                             laserCloudSurfLast->points[minPointInd3].z);

                double s;
                if (DISTORTION)
                  s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                else
                  s = 1.0;
                // 用当前帧的点，和上一帧的对应平面A，B，C构造点到面的距离的残差项，误差项为点到平面的距离，具体见lidarFactor.cpp
                ceres::CostFunction* cost_function =
                    LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                plane_correspondence++;
              }
            }
          }
          printf("data association time %f ms \n", t_data.toc());

          if ((corner_correspondence + plane_correspondence) < 10)
          {
            printf("less correspondence! *************************************************\n");
          }

          TicToc t_solver;
          ceres::Solver::Options options;
          options.linear_solver_type = ceres::DENSE_QR;
          options.max_num_iterations = 4;
          options.minimizer_progress_to_stdout = false;
          ceres::Solver::Summary summary;
          // 基于构建的所有残差项，求解最优的当前帧位姿与上一帧位姿的位姿增量：para_q和para_t
          ceres::Solve(options, &problem, &summary);
          printf("solver time %f ms \n", t_solver.toc());
        }
        printf("optimization twice time %f \n", t_opt.toc());

        // 使用雷达坐标系下的位姿增量更新世界坐标系下的当前位姿
        t_w_curr = t_w_curr + q_w_curr * t_last_curr;
        q_w_curr = q_w_curr * q_last_curr;
      }

      TicToc t_pub;

      // publish odometry(世界坐标系下)
      nav_msgs::Odometry laserOdometry;
      laserOdometry.header.frame_id = "/camera_init";
      laserOdometry.child_frame_id = "/laser_odom";
      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = q_w_curr.x();
      laserOdometry.pose.pose.orientation.y = q_w_curr.y();
      laserOdometry.pose.pose.orientation.z = q_w_curr.z();
      laserOdometry.pose.pose.orientation.w = q_w_curr.w();
      laserOdometry.pose.pose.position.x = t_w_curr.x();
      laserOdometry.pose.pose.position.y = t_w_curr.y();
      laserOdometry.pose.pose.position.z = t_w_curr.z();
      pubLaserOdometry.publish(laserOdometry);

      geometry_msgs::PoseStamped laserPose;
      laserPose.header = laserOdometry.header;
      laserPose.pose = laserOdometry.pose.pose;
      laserPath.header.stamp = laserOdometry.header.stamp;
      laserPath.poses.push_back(laserPose);
      laserPath.header.frame_id = "/camera_init";
      pubLaserPath.publish(laserPath);

      // 将特征点云中的点变换到帧尾，LessSharp，LessFlat
      if (0)
      {
        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++)
        {
          TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
        }

        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++)
        {
          TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        }

        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++)
        {
          TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
        }
      }
      
      // 将LessSharp赋给CornerLast， 将LessFlat赋给SurfLast,然后建立kdtree
      pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();
      
      kdtreeCornerLast->setInputCloud(laserCloudCornerLast);  // 更新kdtree的点云
      kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

      if (frameCount % skipFrameNum == 0)
      {
        frameCount = 0;

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/camera";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
      }
      printf("publication time %f ms \n", t_pub.toc());
      printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
      if (t_whole.toc() > 100)
        ROS_WARN("odometry process over 100ms");

      frameCount++;
    }
    rate.sleep();
  }
  return 0;
}