package annotation

import java.io.{DataInput, DataOutput}

import org.datavec.api.writable.{ArrayWritable, Writable}

/**
  * Created by paolo on 24/05/2017.
  */
trait AnnotationDataSet {

  def getAnnotationsByFile(path: String): Seq[Annotation]

}

case class Annotation(x: Int, y: Int, w: Int, h: Int){

}

class SelectiveSearchAnnotation extends AnnotationDataSet{

  var annotations: Map[String, Seq[Annotation]] = null

  def loadFromFile(path: String)
  {
      val source = scala.io.Source.fromFile(path)
      val annotationMap: Map[String, Seq[(String, Annotation)]] = source.getLines().toSeq.map(e => e.split('|')).map(e => ( e(0)+"/"+e(1) , Annotation(e(2).toInt,e(3).toInt,e(4).toInt,e(5).toInt))).groupBy(_._1)
      annotations = annotationMap.mapValues(_.map(_._2))
  }

  def getAnnotationsByFile(path: String): Seq[Annotation] = {
    annotations(path)
  }

}


class MultiNDArrayWritable(val list: Seq[Writable]) extends ArrayWritable {

  override def getDouble(i: Long): Double = throw new UnsupportedOperationException

  override def length(): Long = throw new UnsupportedOperationException

  override def getFloat(i: Long): Float = throw new UnsupportedOperationException

  override def getLong(i: Long): Long = throw new UnsupportedOperationException

  override def getInt(i: Long): Int = throw new UnsupportedOperationException

  override def write(out: DataOutput): Unit = throw new UnsupportedOperationException

  override def readFields(in: DataInput): Unit = throw new UnsupportedOperationException
}
