//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:1.0-RC1"

import scalismo.ui.api.ScalismoUI
import scalismo.statisticalmodel.GaussianProcess
import scalismo.geometry._3D
import scalismo.geometry.EuclideanVector
import scalismo.geometry.Point
import scalismo.geometry.Point3D
import scalismo.common.Field
import scalismo.kernels.DiagonalKernel
import scalismo.kernels.GaussianKernel
import java.awt.Color
import scala.collection.parallel.CollectionConverters._
import scalismo.statisticalmodel.MultivariateNormalDistribution
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector


@main def gpr() =
  println("Gaussian Process Regression")
  val showUI = true

  val boardSize = if showUI then 20 else 10000

  val points: IndexedSeq[Point[_3D]] = (0 until boardSize).flatMap{i => 
    (0 until boardSize).map{j => 
      Point3D(i.toDouble, j.toDouble, boardSize/2.0)
    }
  }.toIndexedSeq

  println(f"Number of points: ${points.length}")

  val kernel = DiagonalKernel(GaussianKernel[_3D](1.0) * 1.0, 3)
  val gp: GaussianProcess[_3D, EuclideanVector[_3D]] = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

  val posPoints: IndexedSeq[Point[_3D]] = IndexedSeq(
    points(0),
    points((boardSize*boardSize/2-boardSize/2).toInt),
  )

  val noises = if showUI then Seq(1.0, 2.0, 10.0) else Seq(1.0)

  val posteriorPoints = noises.map{noise =>

    val pairs = IndexedSeq(
      (posPoints(0), EuclideanVector(0.0, 0.0, boardSize.toDouble), MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 1.0)) ,
      (posPoints(1), EuclideanVector(0.0, 0.0, boardSize.toDouble), MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * noise.toDouble)) ,
    )

    val startTime = System.nanoTime()
    val mean: Field[_3D, EuclideanVector[_3D]] = gp.posteriorMean(pairs)
    val midTime = System.nanoTime()
    val newPoints: IndexedSeq[Point[_3D]] = points.par.map(p => p + mean(p)).toIndexedSeq
    val endTime = System.nanoTime()

    val executionTimePos = (midTime - startTime) / 1e9
    val executionTimeAdd = (endTime - midTime) / 1e9
    println(s"Execution time posterior : ${"%.2f".format(executionTimePos)}s")
    println(s"Execution time adding : ${"%.2f".format(executionTimeAdd)}s")

    newPoints
  }

  if showUI then
    val colors = Seq(Color.RED, Color.GREEN, Color.BLUE)
    val ui = ScalismoUI()
    val showPoints = ui.show(points, "init-points")
    posteriorPoints.zipWithIndex.foreach{ case (posPoints, i) => 
      val showPosPoints = ui.show(posPoints, s"posteriorpoints - ${noises(i)}")
      showPosPoints._1.radius.value = 1.0
      showPosPoints.color = colors(i)
    }
    showPoints._1.radius.value = 1.0

  println("Done")


