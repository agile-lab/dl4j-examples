package deeplogo

import java.io.File
import javax.imageio.ImageIO

import net.{CustomNet, NetInterface}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

/**
  * Created by andlatel on 17/05/2017.
  */
object Main extends App{

  val network: MultiLayerNetwork = new CustomNet().createNet()

  val baseFolder = "d:\\Users\\andlatel\\Desktop\\jpg\\"

  val listFolder = (new File(baseFolder)).listFiles.filter(_.isDirectory).map(_.listFiles().map(files => ))
  val listFiles = listFolder.flatMap(fold => new File(baseFolder+"\\"+fold).listFiles().filter(_.isFile)).map(_.getAbsolutePath)

  def padImage(file: String) = {

    val image = ImageIO.read(new File(""))

    val newImage = resizeImage(image, 320, 240)

    ImageIO.write(newImage, "png", new File("D:\\resize.jpg"))
  }

  import java.awt.image.BufferedImage
  import org.imgscalr.Scalr

  def resizeImage(inputImage: BufferedImage, resultWidth: Int, resultHeight: Int): BufferedImage = { // first get the width and the height of the image
    val originWidth = inputImage.getWidth
    val originHeight = inputImage.getHeight
    // let us check if we have to scale the image
    if (originWidth <= resultWidth && originHeight <= resultHeight) { // we don't have to scale the image, just return the origin
      return inputImage
    }
    // Scale in respect to width or height?
    var scaleMode = Scalr.Mode.AUTOMATIC
    // find out which side is the shortest
    var maxSize = 0
    if (originHeight > originWidth) { // scale to width
      scaleMode = Scalr.Mode.FIT_TO_WIDTH
      maxSize = resultWidth
    }
    else if (originWidth >= originHeight) {
      scaleMode = Scalr.Mode.FIT_TO_HEIGHT
      maxSize = resultHeight
    }
    // Scale the image to given size
    var outputImage = Scalr.resize(inputImage, Scalr.Method.QUALITY, scaleMode, maxSize)
    // okay, now let us check that both sides are fitting to our result scaling
    if (scaleMode.equals(Scalr.Mode.FIT_TO_WIDTH) && outputImage.getHeight > resultHeight) { // the height is too large, resize again
      outputImage = Scalr.resize(outputImage, Scalr.Method.QUALITY, Scalr.Mode.FIT_TO_HEIGHT, resultHeight)
    }
    else if (scaleMode.equals(Scalr.Mode.FIT_TO_HEIGHT) && outputImage.getWidth > resultWidth) { // the width is too large, resize again
      outputImage = Scalr.resize(outputImage, Scalr.Method.QUALITY, Scalr.Mode.FIT_TO_WIDTH, resultWidth)
    }
    // now we have an image that is definitely equal or smaller to the given size
    // Now let us check, which side needs black lines
    var paddingSize = 0
    if (outputImage.getWidth != resultWidth) { // we need padding on the width axis
      paddingSize = (resultWidth - outputImage.getWidth) / 2
    }
    else if (outputImage.getHeight != resultHeight) { // we need padding on the height axis
      paddingSize = (resultHeight - outputImage.getHeight) / 2
    }
    // we need padding?
    if (paddingSize > 0) { // add the padding to the image
      outputImage = Scalr.pad(outputImage, paddingSize)
      // now we have to crop the image because the padding was added to all sides
      var x = 0
      var y = 0
      var width = 0
      var height = 0
      if (outputImage.getWidth > resultWidth) { // set the correct range
        x = paddingSize
        y = 0
        width = outputImage.getWidth - (2 * paddingSize)
        height = outputImage.getHeight
      }
      else if (outputImage.getHeight > resultHeight) {
        x = 0
        y = paddingSize
        width = outputImage.getWidth
        height = outputImage.getHeight - (2 * paddingSize)
      }
      // Crop the image
      if (width > 0 && height > 0) outputImage = Scalr.crop(outputImage, x, y, width, height)
    }
    // flush both images
    inputImage.flush()
    outputImage.flush()
    // return the final image
    outputImage
  }

}
