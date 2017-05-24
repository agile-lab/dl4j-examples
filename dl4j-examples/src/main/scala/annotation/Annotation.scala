package annotation

import java.io.{DataInput, DataOutput}

import org.datavec.api.writable.{ArrayWritable, Writable}

/**
  * Created by paolo on 24/05/2017.
  */
trait Annotation {

}


class MultiNDArrayWritable(val list: List[Writable]) extends ArrayWritable {

  override def getDouble(i: Long): Double = throw new UnsupportedOperationException

  override def length(): Long = throw new UnsupportedOperationException

  override def getFloat(i: Long): Float = throw new UnsupportedOperationException

  override def getLong(i: Long): Long = throw new UnsupportedOperationException

  override def getInt(i: Long): Int = throw new UnsupportedOperationException

  override def write(out: DataOutput): Unit = throw new UnsupportedOperationException

  override def readFields(in: DataInput): Unit = throw new UnsupportedOperationException
}
