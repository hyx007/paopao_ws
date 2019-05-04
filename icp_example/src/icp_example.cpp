#include "icp.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
/*说明：1、本程序是从pcl库ICP源码中节选出来的程序，程序实现在“icp.h”文件中
 *2、为了简洁起见与源码相比省略了很多类型转换与判断的部分，但是保留了PCL ICP算法主要计算函数的实现方式，对更细节感兴趣的同学可自行阅读源码
 *3、本程序同时提供了PCL中ICP的接口，以对比本程序移植没有问题
 *4、本例程参考PCL ICP官方例程：http://pointclouds.org/documentation/tutorials/iterative_closest_point.php
 *5、有问题欢迎联系：yuxiqd@163.com
 */

int main(){

    /*---------------------------生成用于ICP匹配的点云数据----------------------*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

    cloud_in->width    = 5;
    cloud_in->height   = 1;
    cloud_in->is_dense = false;
    cloud_in->points.resize (cloud_in->width * cloud_in->height);
    //生成source点云
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
    {
      cloud_in->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
      cloud_in->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
      cloud_in->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
    }
    //打印source点云数据
    for (size_t i = 0; i < cloud_in->points.size (); ++i) std::cout << "    " <<
        cloud_in->points[i].x << " " << cloud_in->points[i].y << " " <<
        cloud_in->points[i].z << std::endl;
    *cloud_out = *cloud_in;
    std::cout << "size:" << cloud_out->points.size() << std::endl;
//    生成target点云，target点云为在source点云中每个点的x方向+0.7生成，没有旋转，这样x方向平移0.7便为真值,同理y真值为-0.3
    for (size_t i = 0; i < cloud_in->points.size (); ++i) {
      cloud_out->points[i].x = cloud_in->points[i].x + 0.7f;
      cloud_out->points[i].y = cloud_in->points[i].y - 0.3f;
    }
    //打印target点云
    for (size_t i = 0; i < cloud_out->points.size (); ++i)
      std::cout << "    " << cloud_out->points[i].x << " " <<
        cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;

    /*----------------------使用本程序精简的ICP方法-------------------------------*/
    ICP<pcl::PointXYZ> icp;
    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    std::cout << "my icp has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    /*-------------------PCL库中ICP实现方法------------------*/
    std::cout<<"--------------------------------"<<std::endl;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_pcl;
    icp_pcl.setInputSource(cloud_in);
    icp_pcl.setInputTarget(cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final_pcl;
    icp_pcl.align(Final_pcl);
    std::cout << "pcl icp has converged:" << icp_pcl.hasConverged() << " score: " <<
    icp_pcl.getFitnessScore() << std::endl;
    std::cout << icp_pcl.getFinalTransformation() << std::endl;

 return 0;
}
