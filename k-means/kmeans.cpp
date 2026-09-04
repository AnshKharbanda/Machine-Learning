#include<algorithm>
#include<iostream>
#include<vector>
#include<cmath>
#include<stdexcept>
#include<map>
#include<random>



class KMeans{
    private:
    int maxIterations;
    int kClusters;
    double epsilon;

    std::vector<std::vector<double>> centroids;
    std::vector<std::vector<std::vector<double>>>clusters;

    public:

    // constructor
    KMeans(int k,int maxIterations=100,double epsilon=0.001):maxIterations(maxIterations),kClusters(k),epsilon(epsilon){}

    void fit(const std::vector<std::vector<double>>&data){
        if(data.empty()){
            throw std::invalid_argument("data cannot be empty");
        }

        if (kClusters <= 0 || kClusters > data.size()){
            throw std::invalid_argument("Invalid number of clusters");
        }

        for(const auto &point:data){
            if(point.size()!=data[0].size()){
                throw std::invalid_argument("All Points should have same dimensions !");
            }
        }

        clusters.clear();
        centroids.clear();

        
        clusters.resize(kClusters);

        // choosing initial random k centroids

        std::vector<int> indices(data.size());

        for (int i = 0; i < data.size(); ++i) {
            indices[i] = i;
        }

        std::random_device rd;
        std::mt19937 generator(rd());

        std::shuffle(indices.begin(), indices.end(), generator);

        for (int i = 0; i < kClusters; ++i) {
            centroids.push_back(data[indices[i]]);
        }

        for(int iteration=0;iteration<maxIterations;iteration++){
            clearClusters();

            assignClusters(data);

            auto oldCentroids=centroids;

            updateCentroids();

            auto newCentroids=centroids;


            if(hasConverged(oldCentroids,newCentroids))break;

        }
    }

    int predict(const std::vector<double>&point){
        double dist=std::numeric_limits<double>::max();
        int closest=-1;

        for(int i=0;i<centroids.size();i++){
            double currDist=euclidean_distance(centroids[i],point);
            if(currDist<dist){
                dist=currDist;
                closest=i;
            }
        }
        return closest;
    }

    void clearClusters(){
        for(auto &cluster:clusters){
            cluster.clear();
        }
    }

    void updateCentroids(){
        for(int i=0;i<centroids.size();i++){
            for(int j=0;j<centroids[i].size();j++){
                double coordinate=0;
                if(clusters[i].size()==0)continue;
                for(const auto &cPoint:clusters[i]){
                    coordinate+=cPoint[j];
                }
                centroids[i][j]=coordinate/clusters[i].size();
            }
        }
    }

    void assignClusters(const std::vector<std::vector<double>>&data){
        // assign labels

        for(auto &point1:data){
            double currDist=std::numeric_limits<double>::max();
            int closestCentroid=-1;

            for(int i=0;i<centroids.size();i++){
                double dist=euclidean_distance(point1,centroids[i]);
                if(dist<currDist){
                    currDist=dist;
                    closestCentroid=i;
                }
            }
            clusters[closestCentroid].push_back(point1);
        }
    }

    bool hasConverged(const std::vector<std::vector<double>>&oldCentroids,const std::vector<std::vector<double>>&newCentroids){
        for(int i=0;i<centroids.size();i++){
            double distance=euclidean_distance(oldCentroids[i],newCentroids[i]);

            if(distance>epsilon)return false;
        }

        return true;
    }

    double euclidean_distance(const std::vector<double>&p1,const std::vector<double>&p2){
        int dimensions=p1.size();

        double sum=0;

        for(int i=0;i<dimensions;i++){
            double diff=p1[i]-p2[i];
            sum+=diff*diff;
        }

        return std::sqrt(sum);
    }

    std::vector<std::vector<double>> get_centroids(){
        return centroids;
    }
};


int main(){
    std::vector<std::vector<double>> data = {
        // Cluster 1
        {1.0, 1.0},
        {1.5, 2.0},
        {2.0, 1.5},
        {2.5, 2.0},
        {1.0, 2.5},

        // Cluster 2
        {8.0, 8.0},
        {8.5, 9.0},
        {9.0, 8.5},
        {9.5, 9.0},
        {8.0, 9.5},

        // Cluster 3
        {15.0, 1.0},
        {15.5, 2.0},
        {16.0, 1.5},
        {16.5, 2.0},
        {15.0, 2.5}
    };

    KMeans model(3);
    model.fit(data);

    std::vector<std::vector<double>>centroids=model.get_centroids();

    for(int i=0;i<centroids.size();i++){
        std::cout<<"Centroid "<<i<<"{";
        for(auto pt:centroids[i]){
            std::cout<<pt<<",";
        }
        std::cout<<"}\n ";
    }

    std::cout<<"\n";

    std::vector<double> testPoint = {2.0, 2.0};

    int cluster = model.predict(testPoint);

    std::cout << "Point belongs to cluster: " << cluster << "\n";

// Output Expected

// Centroid 0{8.6,8.8,}
// Centroid 1{1.6,1.8,}
// Centroid 2{15.6,1.8,}
 
// Point belongs to cluster: 1

    return 0;
}