package deeplogo

import net.{CustomNet, SimpleNet, Vgg16Net, Vgg9WithBottleneckNet}

/**
  * Created by andlatel on 20/05/2017.
  */
object Main extends App {

  val conf = new ConfigurationImpl()
  val net = new Vgg9WithBottleneckNet(conf).createNet()

  new LogoClassification(net, conf).exec()

  //new Test(conf).regionTest()
}
