import sys
import math

'''
Point class
Data set의 point를 선언한다.
'''
class Point:
    def __init__(self, id, x, y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        ''' label
        ## Not classified: None
        ## Noise: 0
        ## Cluster id: 1 ~ n 
        '''
        self.label = None
    
'''
DBSCAN class
DBSCAN을 구현한다.
'''
class DBSCAN:
    def __init__(self, data, n, eps, minPts):
        self.data = data
        self.n = n
        self.eps = eps
        self.minPts = minPts

    '''
    Get neighbors of point
    주어진 point의 eps를 만족하는 neighbors를 구한다.
    '''
    def get_neighbors(self, point):
        return [p for p in self.data if p != point and get_distance(point, p) <= self.eps]

    '''
    Expand cluster
    Candidate point를 clustering한다.
    '''
    def expand_cluster(self, neighbors, cluster_id):
        for point in neighbors:
            if point.label == 0: ## Point is noise
                point.label = cluster_id
                
            if point.label is None: ## Point is unclassified
                point.label = cluster_id
                next_neighbors = self.get_neighbors(point)
                if len(next_neighbors) >= self.minPts:
                    neighbors.extend(next_neighbors)

    '''
    Clustering
    DBSCAN의 clustering을 진행한다.
    '''
    def clustering(self):
        cluster_id = 1  

        for point in self.data:
            if point.label is not None: ## Point is classified
                continue
                
            ## Get neighbors of point
            neighbors = self.get_neighbors(point)

            if len(neighbors) < self.minPts:
                point.label = 0 ## Set point is Noise
                continue

            ## Point is core point
            point.label = cluster_id
            ## Expand cluster
            self.expand_cluster(neighbors, cluster_id)
            cluster_id += 1
        
        ## Make cluster list from points
        clusters = [[] for _ in range(0, cluster_id-1)]
        for point in self.data:
            if point.label == 0: # Noise
                continue
            
            clusters[point.label - 1].append(point.id)
        
        ## Sort clusters
        clusters.sort(key=len, reverse=True)

        ## Select n clusters
        clusters = clusters[:self.n]

        return clusters
        
'''
Get euclidean distance
Point a와 Point b 사이의 거리를 구한다.
'''
def get_distance(a: Point, b: Point) -> float:
    return math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))

if __name__ == "__main__":
    ## Read argv
    input_file = sys.argv[1]
    n = int(sys.argv[2])        ## Number of clusters
    eps = float(sys.argv[3])    ## Epsilon of DBSCAN
    minPts = float(sys.argv[4]) ## MinPts of DBSCAN

    ## Save file name
    input_file_name = input_file.split(".")[0]

    ## Read file
    data = []
    f = open("data-3/"+input_file, "r")
    for line in f.readlines():
        p = line.split()
        point = Point(p[0], p[1], p[2])
        data.append(point)

    ## Run DBSCAN
    dbscan = DBSCAN(data, n, eps, minPts)

    ## Write file
    for idx, output in enumerate(dbscan.clustering()):
        output_file_name = input_file_name + "_cluster_%d.txt" % idx

        with open("test-3/"+output_file_name, "w") as output_file:
            for i in output:
                output_file.write("%d\n" % i)