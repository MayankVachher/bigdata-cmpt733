# anomaly_detection.py
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator
import builtins

sc = SparkContext()
spark = SparkSession.builder \
        .appName("AnomalyDetection") \
        .getOrCreate()

class AnomalyDetection():

    def readData(self, filename):
        self.rawDF = spark.read.parquet(filename).cache()

    def cat2Num(self, df, indices):
        toReduce = df.select([col('rawFeatures').getItem(i) for i in indices])

        reducers = {}
        for i in range(len(indices)):
            distinct_vals = toReduce.select(toReduce.columns[i]).distinct()
            reducers[indices[i]] = distinct_vals.rdd.flatMap(lambda x: x).collect()

        def replaceArr(arr, reducers, indices):
            fin_arr = []
            for i in range(len(arr)):
                if i in indices:
                    zero_vec = [0.0] * len(reducers[i])
                    pos = reducers[i].index(arr[i])
                    zero_vec[pos] = 1.0
                    fin_arr = fin_arr + zero_vec
                else:
                    fin_arr.append(float(arr[i]))
            # print(fin_arr)
            return fin_arr

        reduceBrdcst = sc.broadcast(reducers)
        indexBrdcst = sc.broadcast(indices)
        replaceArrUDF = udf(lambda x: replaceArr(x, reduceBrdcst.value, indexBrdcst.value), ArrayType(DoubleType()))

        df = df.withColumn('features', replaceArrUDF(col('rawFeatures')))

        return df

    def addScore(self, df):
        clusterCounts = df.groupby(col('prediction')).count().rdd.collectAsMap()
        N_max = builtins.max(list(clusterCounts.values()))
        denom = N_max - builtins.min(list(clusterCounts.values()))

        N_max = sc.broadcast(N_max)
        denom = sc.broadcast(denom)
        cc = sc.broadcast(clusterCounts)

        def scorefunc(x, N_max, denom, cc):
            return (N_max - cc[x]) / denom

        scorefuncUDF = udf(lambda x: scorefunc(x, N_max.value, denom.value, cc.value), DoubleType())

        df = df.withColumn('score', scorefuncUDF(col('prediction')))

        return df


    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly 
        df3 = self.addScore(df2).cache()
        df3.show()    

        return df3.where(df3.score > t)


if __name__ == "__main__":
    ad = AnomalyDetection()
    ad.readData('data/logs-features-sample')
    anomalies = ad.detect(8, 0.97)
    print(anomalies.count())
    anomalies.show()