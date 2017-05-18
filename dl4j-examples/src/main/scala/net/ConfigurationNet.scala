package net

import java.util.Random

import org.deeplearning4j.examples.convolution.AnimalsClassification
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, SubsamplingLayer}
import org.slf4j.LoggerFactory

/**
  * Created by andlatel on 17/05/2017.
  */
trait ConfigurationNet {
  val height = 100
  val width = 100
  val channels = 3
  val numExamples = 8240
  val numLabels = 33
  val batchSize = 64
  val seed = 42
  val rng = new Random(seed)
  val listenerFreq = 5
  val iterations = 1
  val epochs = 100
  val splitTrainTest = 0.75
  val nCores = 4
  val save = false
  val modelType = "alexnet"

  protected def convInit(name: String, in: Int, out: Int, kernel: Array[Int], stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer =
    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build

  protected def conv3x3(name: String, out: Int, bias: Double): ConvolutionLayer = {
    return new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name(name).nOut(out).biasInit(bias).build
  }

  protected def conv5x5(name: String, out: Int, stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    return new ConvolutionLayer.Builder(Array[Int](5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build
  }

  protected def maxPool(name: String, kernel: Array[Int]): SubsamplingLayer = {
    return new SubsamplingLayer.Builder(kernel, Array[Int](2, 2)).name(name).build
  }

  protected def fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution): DenseLayer = {
    return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build
  }
}

