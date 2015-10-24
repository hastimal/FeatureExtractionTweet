package edu.umkc.group1

import edu.umkc.group1.NLPUtils._
import edu.umkc.group1.Utils._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
 * Created by Mayanka on 14-Jul-15.
 * Modified by hastimal on 10/23/2015.
 */
object FeatureVectorFood {

  def main(args: Array[String]) {
    //System.setProperty("hadoop.home.dir", "C:\\Users\\Jordan\\winutils")
    System.setProperty("hadoop.home.dir", "F:\\winutils")
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("FeatureVectorFood").set("spark.driver.memory", "3g").set("spark.executor.memory", "3g")
    val ssc = new StreamingContext(sparkConf, Seconds(2))
    val sc = ssc.sparkContext
    val stopWords = sc.broadcast(loadStopWords("/stopwords.txt")).value
    val labelToNumeric = createLabelMap("data/training/")
    var model: NaiveBayesModel = null
    // Training the data
    val training = sc.wholeTextFiles("data/training/*")
      .map(rawText => createLabeledDocument(rawText, labelToNumeric, stopWords))
    val X_train = tfidfTransformer(training)
    X_train.foreach(vv => println(vv))

    model = NaiveBayes.train(X_train, lambda = 1.0)

    val lines=sc.wholeTextFiles("data/testing/*")
    val data = lines.map(line => {

      val test = createLabeledDocumentTest(line._2, labelToNumeric, stopWords)
      println(test.body)
      test


    })

    val X_test = tfidfTransformerTest(sc, data)

    val predictionAndLabel = model.predict(X_test)
    println("PREDICTION")
    predictionAndLabel.foreach(x => {
      labelToNumeric.foreach { y => if (y._2 == x) {
        println(y._1)
      }
      }
    })

  }


}
