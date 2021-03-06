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
 * @file laserMapping.cpp
 * @brief：根据点云的边缘点和平面点特征进行scan-scan的匹配
 * @input ：cloud, odom, cloud_corner_last, cloud_surf_last
 * @output：map map_path
 */

#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

int frameCount = 0;

// 用于消息同步
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

// 
int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;

const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;

// map中cube的总数量:21 * 21 * 11 = 4851
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;  // 4851
// 记录submap中的有效cube的index，注意submap中cube的最大数量为 5 * 5 * 5 = 125
int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// 存放cube点云特征的数组，数组大小4851，points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

// kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

// mapping估计优化的位姿
double parameters[7] = { 0, 0, 0, 1, 0, 0, 0 };

// mapping估计优化的位姿映射到世界坐标系下
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// mapping计算的位姿与odometry计算的位姿的增量
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

// odometry计算的位姿在世界坐标系下
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

// 消息队列
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

// 降采样
pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

// kdtree搜索
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel; // 投影

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped,
    pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;
nav_msgs::Path laserAfterMappedPath;

// 将上一帧的增量wmap_wodom * 本帧里程计位姿wodom_curr，旨在为本帧Mapping位姿w_curr设置一个初始值
void transformAssociateToMap()
{
  q_w_curr = q_wmap_wodom * q_wodom_curr;
  t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

// 当Mapping的位姿w_curr计算完毕后，更新增量wmap_wodom，旨在为下一次执行transformAssociateToMap函数时做准备
void transformUpdate()
{
  q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
  t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

// 用Mapping的位姿w_curr，将Lidar坐标系下的点变换到world坐标系下
void pointAssociateToMap(PointType const* const pi, PointType* const po)
{
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->intensity = pi->intensity;
}

// pointAssociateToMap的逆变换，即用Mapping的位姿w_curr，将world坐标系下的点变换到Lidar坐标系下
void pointAssociateTobeMapped(PointType const* const pi, PointType* const po)
{
  Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
  po->x = point_curr.x();
  po->y = point_curr.y();
  po->z = point_curr.z();
  po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudCornerLast2)
{
  mBuf.lock();
  cornerLastBuf.push(laserCloudCornerLast2);
  mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudSurfLast2)
{
  mBuf.lock();
  surfLastBuf.push(laserCloudSurfLast2);
  mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  mBuf.lock();
  fullResBuf.push(laserCloudFullRes2);
  mBuf.unlock();
}

// receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry)
{
  mBuf.lock();
  odometryBuf.push(laserOdometry);
  mBuf.unlock();

  // 雷达里程计pub的odom是在世界坐标系下的
  Eigen::Quaterniond q_wodom_curr;
  Eigen::Vector3d t_wodom_curr;
  q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
  q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
  q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
  q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
  t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
  t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
  t_wodom_curr.z() = laserOdometry->pose.pose.position.z;
  
  // 根据上一帧的wmap_wodom位姿增量更新当前的odom位姿
  Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
  Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
  
  // 发布更新后的odom位姿(世界坐标系下)
  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "/camera_init";
  odomAftMapped.child_frame_id = "/aft_mapped";
  odomAftMapped.header.stamp = laserOdometry->header.stamp;
  odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
  odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
  odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
  odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
  odomAftMapped.pose.pose.position.x = t_w_curr.x();
  odomAftMapped.pose.pose.position.y = t_w_curr.y();
  odomAftMapped.pose.pose.position.z = t_w_curr.z();
  pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void process()
{
  while (1)
  {
    // 为了保证LOAM算法的实时性，每次都只处理最新的消息，同时清空以前的消息
    while (!cornerLastBuf.empty() && !surfLastBuf.empty() && !fullResBuf.empty() && !odometryBuf.empty())
    {
      mBuf.lock();
      // odometryBuf只保留一个与cornerLastBuf.front()时间同步的最新消息
      while (!odometryBuf.empty() &&
             odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
        odometryBuf.pop();
      if (odometryBuf.empty())
      {
        mBuf.unlock();
        break;
      }

      // surfLastBuf也如此
      while (!surfLastBuf.empty() &&
             surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
        surfLastBuf.pop();
      if (surfLastBuf.empty())
      {
        mBuf.unlock();
        break;
      }

      // fullResBuf也如此
      while (!fullResBuf.empty() &&
             fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
        fullResBuf.pop();
      if (fullResBuf.empty())
      {
        mBuf.unlock();
        break;
      }

      timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
      timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
      timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
      timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

      if (timeLaserCloudCornerLast != timeLaserOdometry || timeLaserCloudSurfLast != timeLaserOdometry ||
          timeLaserCloudFullRes != timeLaserOdometry)
      {
        printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast,
               timeLaserCloudFullRes, timeLaserOdometry);
        printf("unsync messeage!");
        mBuf.unlock();
        break;
      }
      
      // 同步后的点云
      laserCloudCornerLast->clear();
      pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
      cornerLastBuf.pop();

      laserCloudSurfLast->clear();
      pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
      surfLastBuf.pop();

      laserCloudFullRes->clear();
      pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
      fullResBuf.pop();

      q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
      q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
      q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
      q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
      t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
      t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
      t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
      odometryBuf.pop();

      // 清空cornerLastBuf的历史缓存，为了LOAM的整体实时性
      while (!cornerLastBuf.empty())
      {
        cornerLastBuf.pop();
        printf("drop lidar frame in mapping for real time performance \n");
      }
      mBuf.unlock();
      TicToc t_whole;

      // 上一帧的增量wmap_wodom * 本帧Odometry位姿wodom_curr，旨在为本帧Mapping位姿w_curr设置一个初始值
      transformAssociateToMap();

      TicToc t_shift;
      // 当前帧位置t_w_curr的IJK坐标，LOAM中使用一维数组管理cube，index = i + j * width + k * width * height
      int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth; // 每个cube为50m
      int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
      int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;

      // 由于计算机求余是向下取整，为了不使（-50.0,50.0）求余后都向零偏移
      if (t_w_curr.x() + 25.0 < 0)
        centerCubeI--;
      if (t_w_curr.y() + 25.0 < 0)
        centerCubeJ--;
      if (t_w_curr.z() + 25.0 < 0)
        centerCubeK--;

      // 求取最终的centerCube中心(没有弄懂)
      //调整之后取值范围:3 < centerCubeI < 18， 3 < centerCubeJ < 18, 3 < centerCubeK < 8
      //如果处于下边界，表明地图向负方向延伸的可能性比较大，则循环移位，将数组中心点向上边界调整一个单位
      while (centerCubeI < 3)
      {
        for (int j = 0; j < laserCloudHeight; j++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int i = laserCloudWidth - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; i >= 1; i--)  
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeI++;
        laserCloudCenWidth++;
      }
      //如果处于上边界，表明地图向正方向延伸的可能性比较大，则循环移位，将数组中心点向下边界调整一个单位
      while (centerCubeI >= laserCloudWidth - 3)
      {
        for (int j = 0; j < laserCloudHeight; j++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int i = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; i < laserCloudWidth - 1; i++)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeI--;
        laserCloudCenWidth--;
      }
      
      while (centerCubeJ < 3)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int j = laserCloudHeight - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; j >= 1; j--)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
      }
      while (centerCubeJ >= laserCloudHeight - 3)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int k = 0; k < laserCloudDepth; k++)
          {
            int j = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; j < laserCloudHeight - 1; j++)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
      }

      while (centerCubeK < 3)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int j = 0; j < laserCloudHeight; j++)
          {
            int k = laserCloudDepth - 1;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; k >= 1; k--)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeK++;
        laserCloudCenDepth++;
      }
      while (centerCubeK >= laserCloudDepth - 3)
      {
        for (int i = 0; i < laserCloudWidth; i++)
        {
          for (int j = 0; j < laserCloudHeight; j++)
          {
            int k = 0;
            pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
                laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
                laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
            for (; k < laserCloudDepth - 1; k++)
            {
              laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
              laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                  laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
            }
            laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeCornerPointer;
            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
                laserCloudCubeSurfPointer;
            laserCloudCubeCornerPointer->clear();
            laserCloudCubeSurfPointer->clear();
          }
        }

        centerCubeK--;
        laserCloudCenDepth--;
      }

      // IJ方向正负扩展2个cube，K方向正负扩展1个cube，从而得到5X5X3的submap的index
      int laserCloudValidNum = 0;
      int laserCloudSurroundNum = 0;
      for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
      {
        for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
        {
          for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
          {
            if (i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 &&
                k < laserCloudDepth)  // 如果坐标合法
            {
              // 记录submap中的所有cube的index，记为有效index
              laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
              laserCloudValidNum++;
              laserCloudSurroundInd[laserCloudSurroundNum] =
                  i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
              laserCloudSurroundNum++;
            }
          }
        }
      }
      
      // 将有效index的cube中的点云叠加到一起组成submap的特征点云
      laserCloudCornerFromMap->clear();
      laserCloudSurfFromMap->clear();
      for (int i = 0; i < laserCloudValidNum; i++)
      {
        *laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
        *laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
      }
      int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
      int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();
      
      // 对接收的corner和surf特征点云进行降采样处理
      pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
      downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
      downSizeFilterCorner.filter(*laserCloudCornerStack);
      int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

      pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
      downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
      downSizeFilterSurf.filter(*laserCloudSurfStack);
      int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

      printf("map prepare time %f ms\n", t_shift.toc());
      printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
      if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50) // submap中的特征点过少
      {
        TicToc t_opt;
        TicToc t_tree;
        // 对submap地图进行kdtree处理
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
        printf("build tree time %f ms \n", t_tree.toc());

        for (int iterCount = 0; iterCount < 2; iterCount++)
        {
          // 定义ceres优化器
          // ceres::LossFunction *loss_function = NULL;
          ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
          ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
          ceres::Problem::Options problem_options;
          ceres::Problem problem(problem_options);
          problem.AddParameterBlock(parameters, 4, q_parameterization);
          problem.AddParameterBlock(parameters + 4, 3);

          TicToc t_data;
          int corner_num = 0;
          //对所有的corner点寻找correspondence
          for (int i = 0; i < laserCloudCornerStackNum; i++)
          {
            pointOri = laserCloudCornerStack->points[i];
            // 将lidar坐标系下的特征点投影到map坐标系下
            pointAssociateToMap(&pointOri, &pointSel);
            // 在submap中，寻找距离corner特征点的投影点最近的5个点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            if (pointSearchSqDis[4] < 1.0) //kdtree搜索后的点经过排序后的，所有比较最后一个点就可以
            {
              // 计算这5个邻近点的中心
              std::vector<Eigen::Vector3d> nearCorners;
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++)
              {
                Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                    laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                    laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
              }
              center = center / 5.0;

              // 计算这5个邻近点的协方差矩阵
              Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
              for (int j = 0; j < 5; j++)
              {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
              }

              // 计算协方差矩阵的特征值和特征向量
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

              // 如果这5个点是线性分布，那么最大的特征值会远大于其他的特征值，最大特征值对应的特征向量为线的方向向量
              Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);  
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) 
              {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                // 从中心点沿着方向向量向两端移动0.1m，构造线上的两个点
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                // 然后残差函数的形式就跟Odometry一样了，残差距离即点到线的距离
                ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                corner_num++;
              }
            }
            /*
            else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
            {
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++)
              {
                Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                          laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                          laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                center = center + tmp;
              }
              center = center / 5.0;
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
              problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
            }
            */
          }

          int surf_num = 0;
          for (int i = 0; i < laserCloudSurfStackNum; i++)
          {
            pointOri = laserCloudSurfStack->points[i];
            // 将lidar坐标系下的特征点投影到map坐标系下
            pointAssociateToMap(&pointOri, &pointSel);
            // 选取投影点最邻近的5个点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // 论文中使用的PCA求解平面，此处使用平面方程构建最小二乘，Ax + By + Cz + 1 = 0 ????
            // 构建平面中的点和平面法向量方程，matA0 * norm（A, B, C） = matB0
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (pointSearchSqDis[4] < 1.0)
            {
              for (int j = 0; j < 5; j++)
              {
                matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
                matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
                matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
              }
              // 求解这个最小二乘问题，可得平面的法向量
              Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
              double negative_OA_dot_norm = 1 / norm.norm(); // ?????
              norm.normalize();

              bool planeValid = true;
              for (int j = 0; j < 5; j++)
              {
                // 根据点到平面的距离公式判断估计的平面是否平
                if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
                         norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
                         norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                  planeValid = false;  // 平面没有拟合好，平面“不够平”
                  break;
                }
              }
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              if (planeValid)
              {
                // 构造点到面的距离残差项
                ceres::CostFunction* cost_function =
                    LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                surf_num++;
              }
            }
            /*
            else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
            {
              Eigen::Vector3d center(0, 0, 0);
              for (int j = 0; j < 5; j++)
              {
                Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
                          laserCloudSurfFromMap->points[pointSearchInd[j]].y,
                          laserCloudSurfFromMap->points[pointSearchInd[j]].z);
                center = center + tmp;
              }
              center = center / 5.0;
              Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
              ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
              problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
            }
            */
          }

          // printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
          // printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

          printf("mapping data assosiation time %f ms \n", t_data.toc());

          TicToc t_solver;
          ceres::Solver::Options options;
          options.linear_solver_type = ceres::DENSE_QR;
          options.max_num_iterations = 4;
          options.minimizer_progress_to_stdout = false;
          options.check_gradients = false;
          options.gradient_check_relative_precision = 1e-4;
          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);
          printf("mapping solver time %f ms \n", t_solver.toc());

          // printf("time %f \n", timeLaserOdometry);
          // printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
          // printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1],
          // parameters[2], 	   parameters[4], parameters[5], parameters[6]);
        }
        printf("mapping optimization time %f \n", t_opt.toc());
      }
      else
      {
        ROS_WARN("time Map corner and surf num are not enough");
      }

      // 完成2次特征匹配后，用最后匹配计算出的优化变量w_curr，更新增量wmap_wodom
      transformUpdate();

      TicToc t_add;
      // 下面两个for loop的作用就是将当前帧的特征点云，逐点进行操作：转换到world坐标系并添加到对应位置的cube中
      for (int i = 0; i < laserCloudCornerStackNum; i++)
      {
        // Lidar坐标系转到world坐标系
        pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

        // 计算本次的特征点的IJK坐标，进而确定添加到哪个cube中
        int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
        int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
        int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

        if (pointSel.x + 25.0 < 0)
          cubeI--;
        if (pointSel.y + 25.0 < 0)
          cubeJ--;
        if (pointSel.z + 25.0 < 0)
          cubeK--;
        
        // 要求cube的index是有效的
        if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
            cubeK < laserCloudDepth)
        {
          int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
          laserCloudCornerArray[cubeInd]->push_back(pointSel);
        }
      }

      for (int i = 0; i < laserCloudSurfStackNum; i++)
      {
        pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

        int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
        int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
        int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

        if (pointSel.x + 25.0 < 0)
          cubeI--;
        if (pointSel.y + 25.0 < 0)
          cubeJ--;
        if (pointSel.z + 25.0 < 0)
          cubeK--;

        if (cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 &&
            cubeK < laserCloudDepth)
        {
          int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
          laserCloudSurfArray[cubeInd]->push_back(pointSel);
        }
      }
      printf("add points time %f ms\n", t_add.toc());

      TicToc t_filter;
      // 因为新增加了点云，对之前已经存有点云的cube全部重新进行一次降采样
      for (int i = 0; i < laserCloudValidNum; i++)
      {
        int ind = laserCloudValidInd[i];

        pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
        downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
        downSizeFilterCorner.filter(*tmpCorner);
        laserCloudCornerArray[ind] = tmpCorner;

        pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
        downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
        downSizeFilterSurf.filter(*tmpSurf);
        laserCloudSurfArray[ind] = tmpSurf;
      }
      printf("filter time %f ms \n", t_filter.toc());

      TicToc t_pub;
      // publish surround map for every 5 frame
      if (frameCount % 5 == 0)
      {
        laserCloudSurround->clear();
        for (int i = 0; i < laserCloudSurroundNum; i++)
        {
          int ind = laserCloudSurroundInd[i];
          *laserCloudSurround += *laserCloudCornerArray[ind];
          *laserCloudSurround += *laserCloudSurfArray[ind];
        }

        sensor_msgs::PointCloud2 laserCloudSurround3;
        pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
        laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudSurround3.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(laserCloudSurround3);
      }

      if (frameCount % 20 == 0)
      {
        pcl::PointCloud<PointType> laserCloudMap;
        for (int i = 0; i < 4851; i++)
        {
          laserCloudMap += *laserCloudCornerArray[i];
          laserCloudMap += *laserCloudSurfArray[i];
        }
        sensor_msgs::PointCloud2 laserCloudMsg;
        pcl::toROSMsg(laserCloudMap, laserCloudMsg);
        laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        laserCloudMsg.header.frame_id = "/camera_init";
        pubLaserCloudMap.publish(laserCloudMsg);
      }

      int laserCloudFullResNum = laserCloudFullRes->points.size();
      for (int i = 0; i < laserCloudFullResNum; i++)
      {
        pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
      }

      sensor_msgs::PointCloud2 laserCloudFullRes3;
      pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
      laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      laserCloudFullRes3.header.frame_id = "/camera_init";
      pubLaserCloudFullRes.publish(laserCloudFullRes3);

      printf("mapping pub time %f ms \n", t_pub.toc());

      printf("whole mapping time %f ms +++++\n", t_whole.toc());

      nav_msgs::Odometry odomAftMapped;
      odomAftMapped.header.frame_id = "/camera_init";
      odomAftMapped.child_frame_id = "/aft_mapped";
      odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
      odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
      odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
      odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
      odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
      odomAftMapped.pose.pose.position.x = t_w_curr.x();
      odomAftMapped.pose.pose.position.y = t_w_curr.y();
      odomAftMapped.pose.pose.position.z = t_w_curr.z();
      pubOdomAftMapped.publish(odomAftMapped);

      geometry_msgs::PoseStamped laserAfterMappedPose;
      laserAfterMappedPose.header = odomAftMapped.header;
      laserAfterMappedPose.pose = odomAftMapped.pose.pose;
      laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
      laserAfterMappedPath.header.frame_id = "/camera_init";
      laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
      pubLaserAfterMappedPath.publish(laserAfterMappedPath);

      static tf::TransformBroadcaster br;
      tf::Transform transform;
      tf::Quaternion q;
      transform.setOrigin(tf::Vector3(t_w_curr(0), t_w_curr(1), t_w_curr(2)));
      q.setW(q_w_curr.w());
      q.setX(q_w_curr.x());
      q.setY(q_w_curr.y());
      q.setZ(q_w_curr.z());
      transform.setRotation(q);
      br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

      frameCount++;
    }
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;

  float lineRes = 0;
  float planeRes = 0;
  nh.param<float>("mapping_line_resolution", lineRes, 0.4);
  nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
  printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
  downSizeFilterCorner.setLeafSize(lineRes, lineRes, lineRes);
  downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
  
  // sub消息
  ros::Subscriber subLaserCloudCornerLast =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);
  ros::Subscriber subLaserCloudSurfLast =
      nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);
  ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);
  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);
  
  // pub消息
  pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
  pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
  pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

  for (int i = 0; i < laserCloudNum; i++)
  {
    laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
    laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
  }

  std::thread mapping_process{ process };

  ros::spin();
  return 0;
}