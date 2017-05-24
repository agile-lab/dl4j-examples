import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j

val s = Seq(Seq(1.0,2.0,3.0,4.0,5.0).toArray, Seq(1.0,2.0,3.0,8.0,6.0).toArray, Seq(1.0,9.0,3.0,4.0,7.0).toArray).toArray

val s2 = Seq(Seq(0.0,0.0,1.0,0.0,0.0).toArray).toArray

val a:INDArray = new NDArray(s);
val b:INDArray = new NDArray(s2);

val c:INDArray = new NDArray(a.shape());
val realOutcomeIndex = Nd4j.argMax(a, 1)
realOutcomeIndex.shape()(0)

val c1= realOutcomeIndex.shape()(0)-1
for(i <- 0 to realOutcomeIndex.shape()(0)-1){
  val index  = realOutcomeIndex.getDouble(i,0)
  c.putScalar(i, index.toInt, 1.0)
}

c

c.add(c)

//c

//a.add(b)

