package deeplogo

import java.util.Random

import org.deeplearning4j.examples.convolution.AnimalsClassification
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, SubsamplingLayer}
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by andlatel on 20/05/2017.
  */
trait Configuration {

  val height = 160
  val width = 160
  val channels = 3
  val numExamples = 850//2240
  val numLabels = 8//32a
  val batchSize = 2//16
  val maxPathPerLabels = 100//160
  val seed = 123
  val rng = new Random(seed)
  val listenerFreq = 10
  val iterations = 1
  val epochs = 1
  val splitTrainTest = 0.8
  val nCores = 6
  val save = false
  val modelType = "custom" //LeNet, AlexNet or Custom but you need to fill it out


  def convInit(name: String, in: Int, out: Int, kernel: Array[Int], stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer =
    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build

  def conv3x3(name: String, out: Int, bias: Double): ConvolutionLayer = {
    return new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name(name).nOut(out).biasInit(bias).build
  }

  def conv5x5(name: String, out: Int, stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    return new ConvolutionLayer.Builder(Array[Int](5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build
  }

  def maxPool(name: String, kernel: Array[Int]): SubsamplingLayer = {
    return new SubsamplingLayer.Builder(kernel, Array[Int](2, 2)).name(name).build
  }

  def fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution): DenseLayer = {
    return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build
  }
}
