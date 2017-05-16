package deeplogo

import java.io.File
import java.util.Random

import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.filters.{BalancedPathFilter, RandomPathFilter}
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform._
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.examples.convolution.AnimalsClassification
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.distribution.{Distribution, GaussianDistribution, NormalDistribution}
import org.deeplearning4j.nn.conf.inputs.{InputType, InvalidInputTypeException}
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._

/**
 * Created by paolo on 15/05/2017.
 */
object LogoClassificationS extends App{
  protected val log = LoggerFactory.getLogger(classOf[AnimalsClassification])
  protected var height = 100
  protected var width = 100
  protected var channels = 3
  protected var numExamples = 6500
  protected var numLabels = 33
  protected var batchSize = 150
  protected var seed = 42
  protected var rng = new Random(seed)
  protected var listenerFreq = 1
  protected var iterations = 1
  protected var epochs = 100
  protected var splitTrainTest = 0.8
  protected var nCores = 4
  protected var save = false
  protected var modelType = "custom" // LeNet, AlexNet or Custom but you need to fill it out

  private def convInit(name: String, in: Int, out: Int, kernel: Array[Int], stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer =
    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build

  private def conv3x3(name: String, out: Int, bias: Double): ConvolutionLayer = {
    return new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name(name).nOut(out).biasInit(bias).build
  }

  private def conv5x5(name: String, out: Int, stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    return new ConvolutionLayer.Builder(Array[Int](5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build
  }

  private def maxPool(name: String, kernel: Array[Int]): SubsamplingLayer = {
    return new SubsamplingLayer.Builder(kernel, Array[Int](2, 2)).name(name).build
  }

  private def fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution): DenseLayer = {
    return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build
  }

  def lenetModel: MultiLayerNetwork = {
    /**
      * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
      * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
      **/
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
      .regularization(false).l2(0.00001).activation(Activation.RELU).learningRate(0.0001)
      .weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.RMSPROP).momentum(0.9)
      .list
      .layer(0, convInit("cnn1", channels, 50, Array[Int](5, 5), Array[Int](1, 1), Array[Int](0, 0), 0))
      .layer(1, maxPool("maxpool1", Array[Int](2, 2)))
      .layer(2, conv5x5("cnn2", 100, Array[Int](5, 5), Array[Int](1, 1), 0))
      .layer(3, maxPool("maxool2", Array[Int](2, 2)))
      .layer(4, new DenseLayer.Builder().nOut(500).build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true).pretrain(false).setInputType(InputType.convolutional(height, width, channels)).build
    return new MultiLayerNetwork(conf)
  }

  def alexnetModel: MultiLayerNetwork = {
    /**
      * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
      * and the imagenetExample code referenced.
      * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
      **/
    val nonZeroBias: Double = 1
    val dropOut: Double = 0.5
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed).weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0.0, 0.01)).activation(Activation.RELU).updater(Updater.NESTEROVS).iterations(iterations).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(1e-2).biasLearningRate(1e-2 * 2)
      .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1).lrPolicySteps(100000)
      .regularization(true).l2(5 * 1e-4).momentum(0.9).miniBatch(false)
      .list
      .layer(0, convInit("cnn1", channels, 96, Array[Int](11, 11), Array[Int](4, 4), Array[Int](3, 3), 0)).layer(1, new LocalResponseNormalization.Builder().name("lrn1").build)
      .layer(2, maxPool("maxpool1", Array[Int](3, 3)))
      .layer(3, conv5x5("cnn2", 256, Array[Int](1, 1), Array[Int](2, 2), nonZeroBias))
      .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build)
      .layer(5, maxPool("maxpool2", Array[Int](3, 3)))
      .layer(6, conv3x3("cnn3", 384, 0))
      .layer(7, conv3x3("cnn4", 384, nonZeroBias))
      .layer(8, conv3x3("cnn5", 256, nonZeroBias))
      .layer(9, maxPool("maxpool3", Array[Int](3, 3)))
      .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true).pretrain(false).setInputType(InputType.convolutional(height, width, channels)).build
    return new MultiLayerNetwork(conf)
  }

  /*def customModel: MultiLayerNetwork = {
    val nonZeroBias: Double = 1
    val dropOut: Double = 0.5
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed)
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU)
      .updater(Updater.NESTEROVS)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(0.01).biasLearningRate(0.001 * 2)
      .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1).lrPolicySteps(100000)
      .regularization(true).l2(5 * 1e-4).momentum(0.95).miniBatch(false)
      .list
      .layer(0, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn1").nIn(channels).nOut(32).biasInit(nonZeroBias).activation(Activation.RELU).build)
      .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("maxpool1").build )
      .layer(2, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn3").nOut(32).biasInit(0).activation(Activation.RELU).build)
      .layer(3, new SubsamplingLayer.Builder(PoolingType.AVG, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("avgpool1").build )
      .layer(4, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn4").nOut(64).biasInit(nonZeroBias).build)
      .layer(5, new SubsamplingLayer.Builder(PoolingType.AVG, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("avgpool2").build)
      .layer(6, new DenseLayer.Builder().name("ffn1").nOut(64).biasInit(nonZeroBias).dropOut(dropOut).dist(new GaussianDistribution(0, 0.005)).build)
      .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true)
      .pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build
    return new MultiLayerNetwork(conf)
  }*/

  def customModel: MultiLayerNetwork = {
    val nonZeroBias: Double = 1
    val dropOut: Double = 0.5
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed)
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU)
      .updater(Updater.NESTEROVS)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(0.01).biasLearningRate(0.001 * 2)
      .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1).lrPolicySteps(100000)
      .regularization(true).l2(5 * 1e-4).momentum(0.95).miniBatch(false)
      .list
      .layer(0, new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name("cnn1").nIn(channels).nOut(32).biasInit(nonZeroBias).activation(Activation.RELU).build)
      .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("maxpool1").build )
      .layer(2, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn3").nOut(64).biasInit(0).activation(Activation.RELU).build)
      .layer(3, new BatchNormalization.Builder().name("btcNorm").build)
      .layer(4, new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1), Array[Int](2, 2)).name("cnn4").nOut(64).biasInit(0).activation(Activation.RELU).build)
      .layer(5, new SubsamplingLayer.Builder(PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).name("maxpool1").build )
      .layer(6, fullyConnected("ffn1", 512, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output").nOut(numLabels).activation(Activation.SOFTMAX).build)
      .backprop(true)
      .pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build
    return new MultiLayerNetwork(conf)
  }




  //-------------------------------MAIN--------------------
  log.info("Load data....")
  /** cd
    * Data Setup -> organize and limit data file paths:
    *  - mainPath = path to image files
    *  - fileSplit = define basic dataset split with limits on format
    *  - pathFilter = define additional file load filter to limit size and balance batch content
    * */
  val labelMaker = new ParentPathLabelGenerator
  //val mainPath = new File(System.getProperty("user.dir"),"dl4j-examples/src/main/resources/animals/");
  val mainPath = new File(System.getProperty("user.home"), "data/FlickrLogos-v2/classes/jpg/")
  val fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng)
  val pathFilter = new RandomPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS, numExamples)
  /**
    * Data Setup -> train test split
    *  - inputSplit = define train and test split
    **/
  val inputSplit: Array[InputSplit] = fileSplit.sample(pathFilter, numExamples * (splitTrainTest), numExamples * (1 - splitTrainTest))
  val fullData = fileSplit.sample(pathFilter,1)(0)
  val trainData = inputSplit(0)
  val testData = inputSplit(1)
  /**
    * Data Setup -> transformation
    *  - Transform = how to tranform images and generate large dataset to train on
    **/
  val flipTransform1 = new FlipImageTransform(rng)
  val flipTransform2 = new FlipImageTransform(new Random(123))
  val warpTransform = new WarpImageTransform(rng, 42)
  val rotateTransform = new RotateImageTransform(new Random(123), 36f)
  //        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);

  val transforms = Seq(flipTransform1, warpTransform, flipTransform2, rotateTransform)

  //ImageTransform transform = new MultiImageTransform(rng, new CropImageTransform(10), new FlipImageTransform(), new ScaleImageTransform(10), new WarpImageTransform(10));

  /**
    * Data Setup -> normalization
    *  - how to normalize images and generate large dataset to train on
    **/
  val scaler = new ImagePreProcessingScaler(0, 1)
  log.info("Build model....")
  // Uncomment below to try AlexNet. Note change height and width to at least 100
  // MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

  val network = modelType match {
    case "LeNet" => lenetModel
    case "AlexNet" => alexnetModel
    case "custom" => customModel
    case _ =>
      throw new InvalidInputTypeException("Incorrect model provided.")
  }
  network.init()
  val uiServer = UIServer.getInstance()
  //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.

  val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later
  //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
  uiServer.attach(statsStorage)
  //Then add the StatsListener to collect this information from the network, as it trains

  val listeners = Seq(new StatsListener(statsStorage), new ScoreIterationListener(listenerFreq))
  network.setListeners(listeners)
  /**
    * Data Setup -> define how to load data into net:
    *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
    *  - dataIter = a generator that only loads one batch at a time into memory to save memory
    *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
    **/
  val recordReader = new ImageRecordReader(height, width, channels, labelMaker)

  log.info("Train model....")

  // Train without transformations
  recordReader.initialize(trainData, null)



  var dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
  val fullDataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)



  log.info("start scaler")
  scaler.fit(fullDataIter)
  log.info("end scaler")
  //scaler.fit(dataIter)
  dataIter.setPreProcessor(scaler)
  var trainIter = new MultipleEpochsIterator(epochs, dataIter)
  log.info("start fit")
  network.fit(trainIter)
  log.info("end fit")

  import org.datavec.api.split.FileSplit



  // Train with transformations
  for (transform <- transforms) {
    System.out.print("\nTraining on transformation: " + transform.getClass.toString + "\n\n")
    recordReader.initialize(trainData, transform)
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)
    trainIter = new MultipleEpochsIterator(epochs, dataIter)
    network.fit(trainIter)
  }



  log.info("Evaluate model....")
  recordReader.initialize(testData)
  val dataIterToEval = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
  //scaler.fit(dataIter)
  dataIterToEval.setPreProcessor(scaler)
  val eval = network.evaluate(dataIterToEval)
  log.info(eval.stats(true))
  // Example on how to get predict results with trained model
  dataIter.reset()

  val testDataSet = dataIter.next
  val expectedResult = testDataSet.getLabelName(0)
  val predict = network.predict(testDataSet)
  val modelResult = predict.get(0)
  System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n")
  if (save) {
    log.info("Save model....")
    val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
    ModelSerializer.writeModel(network, basePath + "model.bin", true)
  }
  log.info("****************Example finished********************")
}
