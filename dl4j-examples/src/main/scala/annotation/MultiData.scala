package annotation
import java.lang

import org.nd4j.linalg.api.rng
import org.nd4j.linalg.dataset
import com.google.common.base.Function
import org.jfree.data.general.Dataset
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.indexing.conditions.Condition
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable
import java.util
import java.util.Random

  /**
    * Created by andlatel on 24/05/2017.
    */


class MultiData extends DataSet{
    val list: Seq[DataSet]

    override def setFeatures(features: INDArray): Unit = ???

    override def divideBy(num: Int): Unit = ???

    override def batchByNumLabels(): util.List[dataset.DataSet] = ???

    override def batchBy(num: Int): util.List[dataset.DataSet] = ???

    override def asList(): util.List[dataset.DataSet] = ???

    override def getLabels: INDArray = ???

    override def iterator(): util.Iterator[dataset.DataSet] = ???

    override def hasMaskArrays: Boolean = ???

    override def getRange(from: Int, to: Int): DataSet = ???

    override def getExampleMetaData[T <: Serializable](metaDataType: Class[T]): util.List[T] = ???

    override def getExampleMetaData: util.List[Serializable] = ???

    override def normalize(): Unit = ???

    override def sortByLabel(): Unit = ???

    override def id(): String = ???

    override def setExampleMetaData(exampleMetaData: util.List[_ <: Serializable]): Unit = ???

    override def setLabels(labels: INDArray): Unit = ???

    override def exampleSums(): INDArray = ???

    override def numInputs(): Int = ???

    override def getFeaturesMaskArray: INDArray = ???

    override def reshape(rows: Int, cols: Int): dataset.DataSet = ???

    override def setLabelNames(labelNames: util.List[String]): Unit = ???

    override def sortAndBatchByNumLabels(): util.List[dataset.DataSet] = ???

    override def exampleMeans(): INDArray = ???

    override def getLabelsMaskArray: INDArray = ???

    override def filterAndStrip(labels: Array[Int]): Unit = ???

    override def getFeatures: INDArray = ???

    override def exampleMaxs(): INDArray = ???

    override def binarize(): Unit = ???

    override def binarize(cutoff: Double): Unit = ???

    override def numExamples(): Int = ???

    override def getColumnNames: util.List[String] = ???

    override def normalizeZeroMeanZeroUnitVariance(): Unit = ???

    override def scaleMinAndMax(min: Double, max: Double): Unit = ???

    override def splitTestAndTrain(numHoldout: Int, rnd: Random): SplitTestAndTrain = ???

    override def splitTestAndTrain(numHoldout: Int): SplitTestAndTrain = ???

    override def splitTestAndTrain(percentTrain: Double): SplitTestAndTrain = ???

    override def setNewNumberOfLabels(labels: Int): Unit = ???

    override def save(to: OutputStream): Unit = ???

    override def save(to: File): Unit = ???

    override def dataSetBatches(num: Int): util.List[dataset.DataSet] = ???

    override def scale(): Unit = ???

    override def addRow(d: dataset.DataSet, i: Int): Unit = ???

    override def iterateWithMiniBatches(): DataSetIterator = ???

    override def numOutcomes(): Int = ???

    override def getFeatureMatrix: INDArray = ???

    override def setColumnNames(columnNames: util.List[String]): Unit = ???

    override def load(from: InputStream): Unit = ???

    override def load(from: File): Unit = ???

    override def roundToTheNearest(roundTo: Int): Unit = ???

    override def get(i: Int): dataset.DataSet = ???

    override def get(i: Array[Int]): dataset.DataSet = ???

    override def getLabelName(idx: Int): String = ???

    override def setOutcome(example: Int, label: Int): Unit = ???

    override def copy(): dataset.DataSet = ???

    override def setFeaturesMaskArray(inputMask: INDArray): Unit = ???

    override def getLabelNames: util.List[String] = ???

    override def getLabelNames(idxs: INDArray): util.List[String] = ???

    override def outcome(): Int = ???

    override def validate(): Unit = ???

    override def getLabelNamesList: util.List[String] = ???

    override def apply(condition: Condition, function: Function[Number, Number]): Unit = ???

    override def filterBy(labels: Array[Int]): dataset.DataSet = ???

    override def sample(numSamples: Int): dataset.DataSet = ???

    override def sample(numSamples: Int, rng: rng.Random): dataset.DataSet = ???

    override def sample(numSamples: Int, withReplacement: Boolean): dataset.DataSet = ???

    override def sample(numSamples: Int, rng: rng.Random, withReplacement: Boolean): dataset.DataSet = ???

    override def addFeatureVector(toAdd: INDArray): Unit = ???

    override def addFeatureVector(feature: INDArray, example: Int): Unit = ???

    override def squishToRange(min: Double, max: Double): Unit = ???

    override def setLabelsMaskArray(labelsMask: INDArray): Unit = ???

    override def labelCounts(): util.Map[Integer, lang.Double] = ???

    override def shuffle(): Unit = ???

    override def multiplyBy(num: Double): Unit = ???

    def add(elem: DataSet) = {
      this.list
    }
  }


