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

#include <cmath>
#include <vector>
#include <string>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>


#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

using std::atan2;
using std::cos;
using std::sin;

// 激光雷达参数
const double scanPeriod = 0.1;  // 扫描周期 10Hz = 0.1s
int N_SCANS = 0;                // 雷达线束：16，32 or 64

// 初始化控制变量
const int systemDelay = 0;
int systemInitCount = 0;
bool systemInited = false;

// 特征提取相关参数
float cloudCurvature[400000];     // 点云曲率
int cloudSortInd[400000];         // 曲率点对应的序号
int cloudNeighborPicked[400000];  // 是否筛选过，0-未筛选过，1-筛选过
int cloudLabel[400000];  // 分类标号：2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小

bool comp(int i, int j)
{
  return (cloudCurvature[i] < cloudCurvature[j]);
}

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;  // 按照线束pub点云

bool PUB_EACH_LINE = false;
double MINIMUM_RANGE = 0.1;  // 滤波范围

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT>& cloud_in, pcl::PointCloud<PointT>& cloud_out, float thres)
{
  if (&cloud_in != &cloud_out)
  {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  size_t j = 0;

  for (size_t i = 0; i < cloud_in.points.size(); ++i)
  {
    if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y +
        cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
      continue;
    cloud_out.points[j] = cloud_in.points[i];
    j++;
  }
  if (j != cloud_in.points.size())
  {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<uint32_t>(j);
  cloud_out.is_dense = true;
}

/**
 * @brief 主要完成以下几件事：
 * @brief 1. 点云预处理：去除NAN点，过滤掉范围内的点
 * @brief 2. 计算点云中起始点和终止点的水平角
 * @brief 3. 计算点云中每个点的垂直角，并划入不同SCAN中，将线束号和获取的相对时间纪录到intensity中
 * @brief 4. 对每个SCAN计算每个点(去除最左最右边上的5个点)的曲率
 * @brief 5. 对曲率进行排序，提取点云中的边缘点和平面点
 * @brief 6. 发布ros消息
 * @param laserCloudMsg
 */
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  if (!systemInited)
  {
    systemInitCount++;
    if (systemInitCount >= systemDelay)
    {
      systemInited = true;
    }
    else
      return;
  }

  TicToc t_whole;
  TicToc t_prepare;

  // 记录每个scan有曲率的点的开始和结束索引
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);

  // 1. 点云预处理：去除NAN点，过滤掉范围内的点
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);  // 去除NAN点
  removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);  // 去除范围内点

  // 2. 计算点云中起始点和终止点的水平角
  int cloudSize = laserCloudIn.points.size();
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;
  // 起始角与终止角异常修正，差值控制在(PI,3*PI)范围
  if (endOri - startOri > 3 * M_PI)
  {
    endOri -= 2 * M_PI;
  }
  else if (endOri - startOri < M_PI)
  {
    endOri += 2 * M_PI;
  }
  
  // 3. 计算点云中每个点的垂直角，根据水平角划入Scan
  bool halfPassed = false;  // lidar扫描线是否旋转过半
  int count = cloudSize;  // 统计除去雷达垂直角范围外的点后的点数
  PointType point;
  std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++)
  {
    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;

    // 计算垂直角
    float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
    int scanID = 0;
    if (N_SCANS == 16)
    {
      scanID = int((angle + 15) / 2 + 0.5);
      if (scanID > (N_SCANS - 1) || scanID < 0)
      {
        count--;
        continue;
      }
    }
    else if (N_SCANS == 32)
    {
      scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
      if (scanID > (N_SCANS - 1) || scanID < 0)
      {
        count--;
        continue;
      }
    }
    else if (N_SCANS == 64)
    {
      if (angle >= -8.83)
        scanID = int((2 - angle) * 3.0 + 0.5);
      else
        scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

      // use [0 50]  > 50 remove outlies
      if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
      {
        count--;
        continue;
      }
    }
    else
    {
      printf("wrong scan number\n");
      ROS_BREAK();
    }
    // printf("angle %f scanID %d \n", angle, scanID);

    // 计算旋转角
    float ori = -atan2(point.y, point.x);
    // 根据旋转角是否过半选择与起始角做差值还是终止角做差值
    if (!halfPassed) 
    {
      //确保-pi/2 < ori - startOri < 3*pi/2
      if (ori < startOri - M_PI / 2)
      {
        ori += 2 * M_PI;
      }
      else if (ori > startOri + M_PI * 3 / 2)
      {
        ori -= 2 * M_PI;
      }
      // 旋转过半，旋转角超过180度
      if (ori - startOri > M_PI)
      {
        halfPassed = true;
      }
    }
    else
    {
      ori += 2 * M_PI;
      //确保-3*pi/2 < ori - endOri < pi/2
      if (ori < endOri - M_PI * 3 / 2)
      {
        ori += 2 * M_PI;
      }
      else if (ori > endOri + M_PI / 2)
      {
        ori -= 2 * M_PI;
      }
    }
    float relTime = (ori - startOri) / (endOri - startOri);
    point.intensity = scanID + scanPeriod * relTime; //点强度=线号+点相对时间
    laserCloudScans[scanID].push_back(point);
  }
  cloudSize = count;
  printf("points size %d \n", cloudSize);

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++)
  {
    scanStartInd[i] = laserCloud->size() + 5; // 每个scan开始计算曲率的点(去除头部5个)
    *laserCloud += laserCloudScans[i];
    scanEndInd[i] = laserCloud->size() - 6;  // 每个scan终止计算曲率的点(去除尾部5个)
  }

  printf("prepare time %f \n", t_prepare.toc());
  
  // 4. 对每个SCAN计算每个点(去除最左最右边上的5个点)的曲率
  for (int i = 5; i < cloudSize - 5; i++)
  {
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x +
                  laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x +
                  laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x +
                  laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y +
                  laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y +
                  laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y +
                  laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z +
                  laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z +
                  laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z +
                  laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ; // 曲率
    cloudSortInd[i] = i; // 曲率排序索引
    cloudNeighborPicked[i] = 0;  // 点有没有被选选择为feature点
    cloudLabel[i] = 0;           // Label 2: corner_sharp
                                 // Label 1: corner_less_sharp, 包含Label 2
                                 // Label -1: surf_flat
                                 // Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样
  }

  // 5. 对曲率进行排序，提取点云中的边缘点和平面点
  TicToc t_pts;
  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;
  float t_q_sort = 0;
  for (int i = 0; i < N_SCANS; i++)  // 按照scan的顺序提取4种特征点
  {
    if (scanEndInd[i] - scanStartInd[i] < 6)  // 如果该scan的点数少于7个点，就跳过
      continue;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    for (int j = 0; j < 6; j++)  // 将该scan分成6小段subscan执行特征检测
    {
      int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;            // subscan的起始index
      int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;  // subscan的结束index

      TicToc t_tmp;
      std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);  // 根据曲率有小到大对subscan的点进行sort
      t_q_sort += t_tmp.toc();

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--)  // 从后往前，即从曲率大的点开始提取corner feature
      {
        int ind = cloudSortInd[k];
        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1)  // 如果该点没有被选择过，并且曲率大于0.1
        {
          largestPickedNum++;
          if (largestPickedNum <= 2)  // 该subscan中曲率最大的前2个点认为是corner_sharp特征点
          {
            cloudLabel[ind] = 2;
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          }
          else if (largestPickedNum <= 20)  // 该subscan中曲率最大的前20个点认为是corner_less_sharp特征点
          {
            cloudLabel[ind] = 1;
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          }
          else
          {
            break;
          }
          cloudNeighborPicked[ind] = 1;  // 标记该点被选择过了

          // 与当前点距离的平方 <= 0.05的点标记为选择过，避免特征点密集分布
          for (int l = 1; l <= 5; l++)
          {
            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
            {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--)
          {
            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
            {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      // 提取surf平面feature，与上述类似，选取该subscan曲率最小的前4个点为surf_flat
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++)
      {
        int ind = cloudSortInd[k];

        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
        {
          cloudLabel[ind] = -1;
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4)
          {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++)
          {
            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
            {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--)
          {
            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
            {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      // 其他的非corner特征点与surf_flat特征点一起组成surf_less_flat特征点
      for (int k = sp; k <= ep; k++)
      {
        if (cloudLabel[k] <= 0)
        {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    // 最后对该scan点云中提取的所有surf_less_flat特征点进行降采样，因为点太多了
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }
  printf("sort q time %f \n", t_q_sort);
  printf("seperate points time %f \n", t_pts.toc());
  
  // 6. 发布ros消息
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera_init";
  pubLaserCloud.publish(laserCloudOutMsg);

  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera_init";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera_init";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera_init";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  // pub each scan
  if (PUB_EACH_LINE)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      sensor_msgs::PointCloud2 scanMsg;
      pcl::toROSMsg(laserCloudScans[i], scanMsg);
      scanMsg.header.stamp = laserCloudMsg->header.stamp;
      scanMsg.header.frame_id = "/camera_init";
      pubEachScan[i].publish(scanMsg);
    }
  }

  printf("scan registration time %f ms *************\n", t_whole.toc());
  if (t_whole.toc() > 100)
    ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  nh.param<int>("scan_line", N_SCANS, 16);

  nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

  printf("scan line number %d \n", N_SCANS);

  if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
  {
    printf("only support velodyne with 16, 32 or 64 scan line!");
    return 0;
  }

  // 订阅节点，接收消息
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

  // 发布节点，接收消息
  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);
  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

  pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

  if (PUB_EACH_LINE)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
      pubEachScan.push_back(tmp);
    }
  }
  ros::spin();

  return 0;
}
