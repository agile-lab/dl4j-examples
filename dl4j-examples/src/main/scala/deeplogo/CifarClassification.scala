package deeplogo

import deeplogo.LogoClassificationS._
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import scala.collection.JavaConversions._

/**
  * Created by andlatel on 17/05/2017.
  */
object CifarClassification extends App{
  val dataSetIterator: CifarDataSetIterator = new CifarDataSetIterator(100, 5000, true);
  System.out.println(dataSetIterator.getLabels)

  import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
  import org.deeplearning4j.nn.conf.layers.OutputLayer
  import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
  import org.deeplearning4j.nn.weights.WeightInit
  import org.nd4j.linalg.activations.Activation
  import org.nd4j.linalg.lossfunctions.LossFunctions


  val epochs = 200

  val layer0: ConvolutionLayer = new ConvolutionLayer.Builder(5, 5).nIn(3).nOut(16).stride(1, 1).padding(2, 2).weightInit(WeightInit.XAVIER).name("First convolution layer").activation(Activation.RELU).build

  val layer1: SubsamplingLayer = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).name("First subsampling layer").build

  val layer2: ConvolutionLayer = new ConvolutionLayer.Builder(5, 5).nOut(20).stride(1, 1).padding(2, 2).weightInit(WeightInit.XAVIER).name("Second convolution layer").activation(Activation.RELU).build

  val layer3: SubsamplingLayer = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).name("Second subsampling layer").build

  val layer4: ConvolutionLayer = new ConvolutionLayer.Builder(5, 5).nOut(20).stride(1, 1).padding(2, 2).weightInit(WeightInit.XAVIER).name("Third convolution layer").activation(Activation.RELU).build

  val layer5: SubsamplingLayer = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).name("Third subsampling layer").build

  val layer6: OutputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).name("Output").nOut(10).build

  val configuration:MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .iterations(1)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(0.0001)
    .regularization(true).l2(0.0004)
    .updater(Updater.NESTEROVS)
    .regularization(true).l2(5 * 1e-5).momentum(0.80)
    .list()
    .layer(0, layer0)
    .layer(1, layer1)
    .layer(2, layer2)
    .layer(3, layer3)
    .layer(4, layer4)
    .layer(5, layer5)
    .layer(6, layer6)
    .pretrain(false)
    .backprop(true)
    .setInputType(InputType.convolutional(32,32,3))
    .build();

  import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

  /*val statsStorage = new InMemoryStatsStorage
  val listeners = Seq(new StatsListener(statsStorage), new ScoreIterationListener(5))
  network.setListeners(listeners)*/

  val network = new MultiLayerNetwork(configuration)
  var trainIter = new MultipleEpochsIterator(epochs, dataSetIterator)
  network.fit(trainIter)

  val layers = network.getLayers
  var totalNumParams = 0
  var i = 0
  while ( i < layers.length) {
    System.out.println("Number of parameters in layer " + i + ": " + layers(i).numParams)
    totalNumParams += layers(i).numParams
    i += 1
  }
  System.out.println("Total number of network parameters: " + totalNumParams)

  import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator

  val evaluation: Evaluation = network.evaluate(new CifarDataSetIterator(2, 500, false))
  System.out.println(evaluation.stats)


}
