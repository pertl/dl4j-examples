package org.deeplearning4j.examples.convolution

import java.util.Random

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.collection.mutable.ListBuffer

object VAEAnomalyDetection {
  def main(args: Array[String]): Unit = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.ADAGRAD)
      .activation(Activation.RELU)
      .learningRate(0.05)
      .regularization(true).l2(0.0001)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(784).nOut(250).build())
      .layer(1, new DenseLayer.Builder().nIn(250).nOut(10).build())
      .layer(2, new DenseLayer.Builder().nIn(10).nOut(250).build())
      .layer(3, new OutputLayer.Builder().nIn(250).nOut(784).lossFunction(LossFunction.MSE).build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.setListeners(new ScoreIterationListener(100))

    val featuresTrain = new ListBuffer[INDArray]()
    val featuresTest = new ListBuffer[INDArray]()
    val labelsTest = new ListBuffer[INDArray]()

    val random = new Random(12345)
    import scala.collection.JavaConversions.asScalaIterator
    for (dataset <- new MnistDataSetIterator(100, 50000, false)) {
      val split = dataset.splitTestAndTrain(80, random)
      featuresTrain.append(split.getTrain.getFeatureMatrix)
      featuresTest.append(split.getTest.getFeatureMatrix)
      labelsTest.append(Nd4j.argMax(split.getTest.getLabels, 1))
    }
    for (epoch <- 1 to 30) {
      for (data <- featuresTrain) {
        net.fit(data, data)
      }
      println(s"epoch #$epoch completed")
    }
  }
}
