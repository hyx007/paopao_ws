#ifndef ICP_H
#define ICP_H

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/registration/boost.h>
#include <pcl/correspondence.h>
template <typename PointT>
class ICP 
{
public:
	typedef pcl::PointCloud<PointT> PointCloudT;
    typedef boost::shared_ptr<pcl::PointCloud<PointT> > PointCloudTPtr;
	typedef boost::shared_ptr<PointT const> PointCloudTConstPtr;
	typedef typename pcl::search::KdTree<PointT>::Ptr KdTreePtr;
	typename pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria_;

public:
	ICP()
        : use_umeyama_(true)
        , nr_iterations_(0)
        , max_iterations_ (10)
        , min_number_correspondences_ (3)
        , euclidean_fitness_epsilon_(-std::numeric_limits<double>::max ())
        , transformation_epsilon_ (0.0)
        , corr_dist_threshold_ (std::sqrt (std::numeric_limits<double>::max ()))
        , transformation_rotation_epsilon_(0.0)
        , transformation_(Eigen::Matrix4f::Identity())
        , use_reciprocal_correspondence_(false) {
        correspondences_.reset(new pcl::Correspondences);
        source_tree_.reset(new pcl::search::KdTree<PointT>);
		target_tree_.reset(new pcl::search::KdTree<PointT>);
		convergence_criteria_.reset(new pcl::registration::DefaultConvergenceCriteria<float>(nr_iterations_, transformation_, *correspondences_));
	};
    //设置最大允许迭代次数，超过此次数退出迭代
	inline void setMaximumIterations(const int nr_iterations) { max_iterations_ = nr_iterations; }
    //设置source点云
    inline void setInputSource(const PointCloudTPtr& cloud) {
		source_cloud_ = cloud;
	}
    //设置双向搜索，找近邻时需要同时满足source与target的近邻点一一对应
    inline void setUseReciprocalCorrespondences (bool use_reciprocal_correspondence) {
      use_reciprocal_correspondence_ = use_reciprocal_correspondence;
    }
    //设置target点云
    inline void setInputTarget(const PointCloudTPtr& cloud) {
		target_cloud_ = cloud;
        target_tree_->setInputCloud(target_cloud_);
	}
    //icp计算入口函数
	void align(PointCloudT& output, const Eigen::Matrix4f& guess = Eigen::Matrix4f::Identity());
    //SVD分解计算ICP
	void estimateRigidTransformationSVD(const pcl::PointCloud<PointT>& cloud_src,
										const pcl::PointCloud<PointT>& cloud_tgt,
										const pcl::Correspondences& correspondences,
                                        Eigen::Matrix4f& transformation_matrix);
	void getTransformationFromCorrelation(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
										  const Eigen::Matrix<float, 4, 1>& centroid_src,
										  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean,
										  const Eigen::Matrix<float, 4, 1>& centroid_tgt,
										  Eigen::Matrix4f& transformation_matrix) const;
    void demeanPointCloud (pcl::ConstCloudIterator<PointT> &cloud_iterator,
                           const Eigen::Matrix<float, 4, 1> &centroid,
                           Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &cloud_out,
                           int npts=0);
    //利用kdtree寻找近邻，确定对应关系
    void determineCorrespondences(const PointCloudTPtr& source_cloud, pcl::Correspondences& correspondences,
		double max_distance);
    //双向搜索确定对应关系
    void determineReciprocalCorrespondences(const PointCloudTPtr& source_cloud, pcl::Correspondences& correspondences,
		double max_distance);
    //设定前后两次迭代的距离差，每次迭代的距离是source与target点云对应点对的欧氏距离均值
    inline void setEuclideanFitnessEpsilon (double epsilon) { euclidean_fitness_epsilon_ = epsilon; }
    //设定前后两次迭代的平移距离阈值
    inline void setTransformationEpsilon (double epsilon) { transformation_epsilon_ = epsilon; }
    //设定前后两次迭代的旋转阈值
    inline void setTransformationRotationEpsilon (double epsilon) { transformation_rotation_epsilon_ = epsilon; }

    void computeTransformation(PointCloudT& output, const Eigen::Matrix4f& guess);
    unsigned int compute3DCentroid(pcl::ConstCloudIterator<PointT>& cloud_iterator,Eigen::Matrix<float, 4, 1>& centroid);
    inline double getFitnessScore (double max_range = std::numeric_limits<double>::max ());
    inline bool hasConverged () { return (converged_); }
    inline Eigen::Matrix4f getFinalTransformation () { return (final_transformation_); }
    void transformCloud (const PointCloudT &input,PointCloudT &output,const Eigen::Matrix4f &transform);
private:
	int max_iterations_;
	bool converged_;
	bool use_reciprocal_correspondence_;
    bool use_umeyama_;
	int nr_iterations_;
    int min_number_correspondences_;

    double euclidean_fitness_epsilon_;
    double transformation_epsilon_;
    double corr_dist_threshold_;
    double transformation_rotation_epsilon_;
    pcl::CorrespondencesPtr correspondences_;

	KdTreePtr source_tree_;
	KdTreePtr target_tree_;
    PointCloudTPtr source_cloud_,target_cloud_;
	Eigen::Matrix4f final_transformation_;
	Eigen::Matrix4f transformation_;
	Eigen::Matrix4f previous_transformation_;
};
/*函数名称：align
**函数功能：ICP算法运行的入口函数，很多同学不知道的是，其实该函数支持输入ICP的初始估计矩阵，guess
*/
template <typename PointT>
void ICP<PointT>::align(PointCloudT& output, const Eigen::Matrix4f& guess) {
	if (!source_cloud_|| !target_cloud_)
		return;
    std::cout<<"source_cloud_ size:"<<source_cloud_->size()<<std::endl;
    std::cout<<"target_cloud_ size:"<<target_cloud_->size()<<std::endl;

	converged_ = false;
	final_transformation_ = transformation_ = previous_transformation_ = Eigen::Matrix4f::Identity();
	computeTransformation(output, guess);
}
/*函数名称：computeTransformation
**函数功能：ICP算法主要流程框架
*/
template <typename PointT>
void ICP<PointT>::computeTransformation(
	PointCloudT& output, const Eigen::Matrix4f& guess) {

	PointCloudTPtr input_transformed(new PointCloudT);
	nr_iterations_ = 0;
	converged_ = false;
	final_transformation_ = guess;

    //如果给定ICP的初始估计矩阵guess，则先要将source点云根据guess变换转换成新的点云
	if (guess != Eigen::Matrix4f::Identity()) {
		input_transformed->resize(source_cloud_->size());
        pcl::transformPointCloud(*source_cloud_,*input_transformed, guess);
	}
	else
		*input_transformed = *source_cloud_;
	transformation_ = Eigen::Matrix4f::Identity();

    //设置ICP迭代收敛条件
	convergence_criteria_->setMaximumIterations(max_iterations_);
	convergence_criteria_->setRelativeMSE(euclidean_fitness_epsilon_);
	convergence_criteria_->setTranslationThreshold(transformation_epsilon_);
	if (transformation_rotation_epsilon_ > 0)
		convergence_criteria_->setRotationThreshold(transformation_rotation_epsilon_);
	else
		convergence_criteria_->setRotationThreshold(1.0 - transformation_epsilon_);
//ICP迭代计算部分
    do {
		previous_transformation_ = transformation_;
        //确立对应关系
		if (use_reciprocal_correspondence_)
			determineReciprocalCorrespondences(input_transformed ,*correspondences_, corr_dist_threshold_);
		else
			determineCorrespondences(input_transformed ,*correspondences_, corr_dist_threshold_);

		size_t cnt = correspondences_->size();
        //判断是否有足够的点对计算icp,默认最少3对
		if (static_cast<int> (cnt) < min_number_correspondences_) {
			PCL_ERROR("Not enough correspondences found. Relax your threshold parameters.\n");
			convergence_criteria_->setConvergenceState(pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES);
			converged_ = false;
			break;
		}
        //SVD分解求解变换矩阵
        estimateRigidTransformationSVD(*input_transformed, *target_cloud_, *correspondences_, transformation_);
        //使用SVD分解得到变换矩阵transformation_再次转换点云
        pcl::transformPointCloud(*input_transformed, *input_transformed, transformation_);
        //根据连乘法则，计算最终的变换矩阵
		final_transformation_ = transformation_ * final_transformation_;
		++nr_iterations_;

        //更新收敛状态，确定是否收敛,如果没有收敛则继续迭代
		converged_ = static_cast<bool> ((*convergence_criteria_));
	} while (!converged_);
	// Transform the input cloud using the final transformation
	PCL_DEBUG("Transformation is:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n",
		final_transformation_(0, 0), final_transformation_(0, 1), final_transformation_(0, 2), final_transformation_(0, 3),
		final_transformation_(1, 0), final_transformation_(1, 1), final_transformation_(1, 2), final_transformation_(1, 3),
		final_transformation_(2, 0), final_transformation_(2, 1), final_transformation_(2, 2), final_transformation_(2, 3),
		final_transformation_(3, 0), final_transformation_(3, 1), final_transformation_(3, 2), final_transformation_(3, 3));

	output = *source_cloud_;
    //将source点云根据计算的变换矩阵转换，得到最终的输出点云
    pcl::transformPointCloud(*source_cloud_, output, final_transformation_);
}
/*函数名称：estimateRigidTransformationSVD
**函数功能：SVD分解主要框架，这里可以选择使用Eigen自带的求解umeyama方法计算，也可以使用pcl自己的求解方法，默认使用umeyama
*/
template <typename PointT>
void ICP<PointT>::estimateRigidTransformationSVD(
	const pcl::PointCloud<PointT>& cloud_src,
	const pcl::PointCloud<PointT>& cloud_tgt,
	const pcl::Correspondences& correspondences,
    Eigen::Matrix4f& transformation_matrix){
	pcl::ConstCloudIterator<PointT> source_it(cloud_src, correspondences, true);
	pcl::ConstCloudIterator<PointT> target_it(cloud_tgt, correspondences, false);
	// Convert to Eigen format
	const int npts = static_cast <int> (source_it.size());
	//用EIGEN自带的方法求解
	if (use_umeyama_) {
		Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_src(3, npts);
		Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_tgt(3, npts);
		for (int i = 0; i < npts; ++i) {
			cloud_src(0, i) = source_it->x;
			cloud_src(1, i) = source_it->y;
			cloud_src(2, i) = source_it->z;
			++source_it;

			cloud_tgt(0, i) = target_it->x;
			cloud_tgt(1, i) = target_it->y;
			cloud_tgt(2, i) = target_it->z;
			++target_it;
		}
		// Call Umeyama directly from Eigen (PCL patched version until Eigen is released)
		transformation_matrix = pcl::umeyama(cloud_src, cloud_tgt, false);
	}
	else {
		source_it.reset(); target_it.reset();
		transformation_matrix.setIdentity();
		Eigen::Matrix<float, 4, 1> centroid_src, centroid_tgt;
		// Estimate the centroids of source, target
		compute3DCentroid(source_it, centroid_src);
		compute3DCentroid(target_it, centroid_tgt);
		source_it.reset(); target_it.reset();
		// Subtract the centroids from source, target
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> cloud_src_demean, cloud_tgt_demean;
		demeanPointCloud(source_it, centroid_src, cloud_src_demean);
		demeanPointCloud(target_it, centroid_tgt, cloud_tgt_demean);
		getTransformationFromCorrelation(cloud_src_demean, centroid_src, cloud_tgt_demean, centroid_tgt, transformation_matrix);
	}
}
/*函数名称：getTransformationFromCorrelation
**函数功能：SVD分解的核心计算部分，即H=source * target',对H进行SVD分解得到矩阵UV,最优旋转矩阵即为R=U*V'
*/
template <typename PointT>
void ICP<PointT>::getTransformationFromCorrelation(
	const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
	const Eigen::Matrix<float, 4, 1>& centroid_src,
	const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean,
	const Eigen::Matrix<float, 4, 1>& centroid_tgt,
	Eigen::Matrix4f& transformation_matrix) const {
	transformation_matrix.setIdentity();

	// Assemble the correlation matrix H = source * target'
	Eigen::Matrix<float, 3, 3> H = (cloud_src_demean * cloud_tgt_demean.transpose()).topLeftCorner(3, 3);

	// Compute the Singular Value Decomposition
	Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<float, 3, 3> u = svd.matrixU();
	Eigen::Matrix<float, 3, 3> v = svd.matrixV();
//    std::cout<<"svd.m_singularValues:"<<svd.singularValues()<<std::endl;
	// Compute R = V * U'
	if (u.determinant() * v.determinant() < 0) {
		for (int x = 0; x < 3; ++x)
			v(x, 2) *= -1;
	}
	Eigen::Matrix<float, 3, 3> R = v * u.transpose();
	// Return the correct transformation
	transformation_matrix.topLeftCorner(3, 3) = R;
	const Eigen::Matrix<float, 3, 1> Rc(R * centroid_src.head(3));
	transformation_matrix.block(0, 3, 3, 1) = centroid_tgt.head(3) - Rc;
}
/*函数名称：compute3DCentroid
**函数功能：计算点云的中心点，即均值
*/
template <typename PointT> 
unsigned int ICP<PointT>::compute3DCentroid(pcl::ConstCloudIterator<PointT>& cloud_iterator,
	Eigen::Matrix<float, 4, 1>& centroid) {
	// Initialize to 0
	centroid.setZero();
	unsigned cp = 0;
	while (cloud_iterator.isValid()) {
		//如果该点坐标均为有效值
		if (pcl::isFinite(*cloud_iterator)) {
			centroid[0] += cloud_iterator->x;
			centroid[1] += cloud_iterator->y;
			centroid[2] += cloud_iterator->z;
			++cp;
		}
		++cloud_iterator;
	}
	centroid /= static_cast<float> (cp);
	centroid[3] = 1;
	return (cp);
}
/*函数名称：determineReciprocalCorrespondences
**函数功能：利用双向kdtree搜索，得到对应点对
*/
template <typename PointT> 
void ICP<PointT>::determineReciprocalCorrespondences(
    const PointCloudTPtr& source_cloud,
	pcl::Correspondences& correspondences, 
	double max_distance) {
	double max_dist_sqr = max_distance * max_distance;
	source_tree_->setInputCloud(source_cloud);

	correspondences.resize(source_cloud->points.size());
	std::vector<int> index(1);
	std::vector<float> distance(1);
	std::vector<int> index_reciprocal(1);
	std::vector<float> distance_reciprocal(1);
	pcl::Correspondence corr;
	unsigned int nr_valid_correspondences = 0;
	int target_idx = 0;

	for (int idx = 0; idx < source_cloud->points.size(); ++idx) {
			target_tree_->nearestKSearch(source_cloud->points[idx], 1, index, distance);
			if (distance[0] > max_dist_sqr)
				continue;
			target_idx = index[0];
			source_tree_->nearestKSearch(target_cloud_->points[target_idx], 1, index_reciprocal, distance_reciprocal);
			if (distance_reciprocal[0] > max_dist_sqr || idx != index_reciprocal[0])
				continue;
			corr.index_query = idx;
			corr.index_match = index[0];
			corr.distance = distance[0];
			correspondences[nr_valid_correspondences++] = corr;
		}
	correspondences.resize(nr_valid_correspondences);
}
/*函数名称：determineCorrespondences
**函数功能：单向kdtree搜索，得到对应点对
*/
template <typename PointT>
void ICP<PointT>::determineCorrespondences(
    const PointCloudTPtr& source_cloud,
	pcl::Correspondences& correspondences, 
	double max_distance) {

	double max_dist_sqr = max_distance * max_distance;
	correspondences.resize(source_cloud->size());
	std::vector<int> index(1);
	std::vector<float> distance(1);
	pcl::Correspondence corr;
	unsigned int nr_valid_correspondences = 0;
	for (int idx = 0; idx < source_cloud->points.size(); ++idx) {
		target_tree_->nearestKSearch(source_cloud->points[idx], 1, index, distance);
		if (distance[0] > max_dist_sqr)
			continue;
			corr.index_query = idx;
			corr.index_match = index[0];
			corr.distance = distance[0];
			correspondences[nr_valid_correspondences++] = corr;
		}
    correspondences.resize(nr_valid_correspondences);
}
/*函数名称：demeanPointCloud
**函数功能：根据点云中心得到去中心点云坐标及其迭代器
*/
template <typename PointT>
void ICP<PointT>::demeanPointCloud (pcl::ConstCloudIterator<PointT> &cloud_iterator,
                  const Eigen::Matrix<float, 4, 1> &centroid,
                  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &cloud_out,
                  int npts) {
  // Calculate the number of points if not given
  if (npts == 0)
  {
    while (cloud_iterator.isValid ())
    {
      ++npts;
      ++cloud_iterator;
    }
    cloud_iterator.reset ();
  }

  cloud_out = Eigen::Matrix<float, 4, Eigen::Dynamic>::Zero (4, npts);        // keep the data aligned

  int i = 0;
  while (cloud_iterator.isValid ())
  {
    cloud_out (0, i) = cloud_iterator->x - centroid[0];
    cloud_out (1, i) = cloud_iterator->y - centroid[1];
    cloud_out (2, i) = cloud_iterator->z - centroid[2];
    ++i;
    ++cloud_iterator;
  }
}
/*函数名称：getFitnessScore
**函数功能：计算ICP得分，这里的得分即近邻点云对的距离均值，因此得分越"小"代表得到的变换矩阵精度越高
*/
template <typename PointT>
inline double ICP<PointT>::getFitnessScore (double max_range) {
  double fitness_score = 0.0;

  // Transform the input dataset using the final transformation
  PointCloudT input_transformed;
  pcl::transformPointCloud (*source_cloud_, input_transformed, final_transformation_);

  std::vector<int> nn_indices (1);
  std::vector<float> nn_dists (1);

  // For each point in the source dataset
  int nr = 0;
  for (size_t i = 0; i < input_transformed.points.size (); ++i) {
    // Find its nearest neighbor in the target
    target_tree_->nearestKSearch (input_transformed.points[i], 1, nn_indices, nn_dists);
    // Deal with occlusions (incomplete targets)
    if (nn_dists[0] <= max_range) {
      // Add to the fitness score
      fitness_score += nn_dists[0];
      nr++;
    }
  }

  if (nr > 0)
    return (fitness_score / nr);
  else
    return (std::numeric_limits<double>::max ());
}



#endif // ICP_H
