import re
import operator
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, functions as f
from pyspark.sql.types import StringType, FloatType

conf = SparkConf().setAppName("Entity Resolution")
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)


class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):
        if len(cols) == 0:
            raise Exception("preprocessDF: Empty cols list!")
        if len(cols) == 1:
            df = df.withColumnRenamed(cols[0],'joinKey')
        else:
            df = df.withColumn('joinKey', f.concat(f.col(cols[0]), f.lit(" "), f.col(cols[1])))
            cols = cols[2:]
            while len(cols) > 0:
                df = df.withColumn('joinKey', f.concat(f.col('joinKey'), f.lit(" "), f.col(cols[0])))
                cols = cols[1:]

        def customTokenizer(string, stopWordSet):
            string = string.lower()
            tokens = re.split(r'\W+', string)
            cleaned_string = []
            for x in tokens:
                if x not in stopWordSet and x != "":
                    cleaned_string.append(x)

            return " ".join(cleaned_string)

        stopWordSet = self.stopWordsBC

        tokenize = f.udf(lambda x: customTokenizer(x, stopWordSet), StringType())

        df = df.withColumn('joinKey', tokenize(f.col('joinKey')))

        return df


    def filtering(self, df1, df2):

        df1_id_token = df1.select(f.col('id').alias('id1'),
                                  f.explode(f.split(f.col('joinKey'), "\s+")).alias("token1"))
        df2_id_token = df2.select(f.col('id').alias('id2'),
                                  f.explode(f.split(f.col('joinKey'), "\s+")).alias("token2"))

        df_ids = df1_id_token.join(df2_id_token, (f.col('token1') == f.col('token2')) & \
                                                  (f.col('id1') < f.col('id2'))).distinct()


        candDF = df1.join(df_ids, f.col('id1') == f.col('id')) \
                    .select(
                        f.col('id1'),
                        f.col('joinKey').alias('joinKey1'),
                        f.col('id2')).distinct()

        candDF = df2.join(candDF, f.col('id2') == f.col('id')) \
                    .select(
                        f.col('id1'),
                        f.col('joinKey1'),
                        f.col('id2'),
                        f.col('joinKey').alias('joinKey2')).distinct()

        # df1_id_token = df1.select(f.col('id').alias('id1'),
        #                           f.col('joinKey').alias('joinKey1'))
        #
        # df2_id_token = df2.select(f.col('id').alias('id2'),
        #                           f.col('joinKey').alias('joinKey2'))
        #
        # candDF = df1_id_token.crossJoin(df2_id_token)

        return candDF

    def verification(self, candDF, threshold):

        def jaccard(joinKey1, joinKey2):
            r = set(joinKey1.split())
            s = set(joinKey2.split())

            unionset = r.union(s)
            intersectionset = r.intersection(s)

            jaccard_ind = len(intersectionset) / len(unionset)

            return jaccard_ind

        jaccard_sim = f.udf(jaccard, FloatType())

        resultDF = candDF.withColumn('jaccard', jaccard_sim(f.col('joinKey1'), f.col('joinKey2')))

        resultDF = resultDF.filter(f.col('jaccard') >= threshold)

        return resultDF

    def evaluate(self, result, groundTruth):
        R = set(result)
        A = set(groundTruth)
        T = A.intersection(R)

        precision = len(T) / len(R)

        recall = len(T) / len(A)

        fmeasure = (2 * precision * recall) / (precision + recall)

        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print("Before filtering: %d pairs in total" % (self.df1.count() * self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print("After Filtering: %d pairs left" % (candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print("After Verification: %d similar pairs" % (resultDF.count()))

        return resultDF

    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet("Amazon_Google_perfectMapping_sample") \
        .rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))