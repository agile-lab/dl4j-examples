package annotation

import java.io.File
import java.util

import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.Mat
import org.datavec.api.io.labels.PathLabelGenerator
import org.datavec.api.writable.{IntWritable, Writable}
import org.datavec.common.RecordConverter
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.examples.deeplogo.{MyImageRecordReader, MyNativeImageLoader}
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by paolo on 24/05/2017.
  */
class AnnotatedImageReader(folderPath: String, annotationDataSet: AnnotationDataSet, height: Int, width: Int, channels: Int, labelGenerator: PathLabelGenerator ) extends MyImageRecordReader(height,width,channels,labelGenerator){


  override def next: java.util.List[Writable] = {
    if (iter != null) {
      var ret: java.util.List[Writable] = null
      val image: File = iter.next
      currentFile = image
      if (image.isDirectory) return next
      try
        invokeListeners(image)

        if(image.getAbsolutePath.contains(folderPath)){

        }else{
          throw new Exception("parameter folder path doesn't match with fetched files")
        }

        val subFile = image.getAbsolutePath.substring(folderPath.size)
        val splits = subFile.split('/')
        val subFolder = splits(0)
        val file = splits(1)

        val annotations: Seq[Annotation] = annotationDataSet.getAnnotationsByFile(subFile)

        val myLoader = imageLoader.asInstanceOf[MyNativeImageLoader]

        val mat: Mat = myLoader.asOpenCVMat(image)

        val images = annotations.map(a => {
          val croppedImage = mat.apply(new opencv_core.Rect(a.x, a.y, a.w, a.h))
          myLoader.asMatrix(croppedImage)
        })

        val records: Seq[Writable] = images.map(im => {
          RecordConverter.toRecord(im).get(0)
        })

        val multiImageRecords = new MultiNDArrayWritable(records)

        ret = new java.util.ArrayList[Writable]()
        ret.add(multiImageRecords)

        if (appendLabel)
          ret.add(new IntWritable(labels.indexOf(getLabel(image.getPath))))


      catch {
        case e: Exception => {
          throw new RuntimeException(e)
        }
      }
      return ret
    }
    else if (record != null) {
      hitImage = true
      invokeListeners(record)
      return record
    }
    throw new IllegalStateException("No more elements")
  }

}
