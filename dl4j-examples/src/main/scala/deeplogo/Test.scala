package deeplogo

import java.io.File

import deeplogo.Main.conf
import net.CustomNet
import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by andlatel on 23/05/2017.
  */
class Test(conf: Configuration){
  val log: Logger = LoggerFactory.getLogger(classOf[LogoClassification])
  def test() = {

    log.info("Load model....")

    val labelMaker = new ParentPathLabelGenerator

    val mainPath = new File("d:\\Users\\andlatel\\Desktop\\TrainLogo\\")
    val fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, conf.rng)
    val pathFilter = new BalancedPathFilter(conf.rng, labelMaker, 52, conf.numLabels, 8)

    val inputSplit: Array[InputSplit] = fileSplit.sample(pathFilter, conf.numExamples)
    //val inputSplitTest: Array[InputSplit] = fileSplitTest.sample(pathFilterTest, 2240, 0)
    val trainData = inputSplit(0)


    //val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
    //val network = ModelSerializer.restoreMultiLayerNetwork(basePath + "model.bin", true)
    val network = ModelSerializer.restoreMultiLayerNetwork("d:\\Users\\andlatel\\Desktop\\model.bin", true)


    val testData = inputSplit(0)

    val scaler = new ImagePreProcessingScaler(0, 1)

    log.info("Evaluate model....")
    val recordReader = new ImageRecordReader(conf.height, conf.width, conf.channels, labelMaker)
    recordReader.initialize(testData, null)
    var dataIter: DataSetIterator   = new RecordReaderDataSetIterator(recordReader, conf.batchSize, 1, conf.numLabels)
    dataIter.setPreProcessor(scaler)
    val eval = network.evaluate(dataIter)
    log.info(eval.stats(true))
  }
}
