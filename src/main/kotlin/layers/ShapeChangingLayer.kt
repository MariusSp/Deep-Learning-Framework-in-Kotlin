package layers

import structure.Shape
import structure.Tensor

interface ShapeChangingLayer : Layer {
    val outputShape: Shape
    fun updateWeights(learningRate: Float, optimizer: (Float, ArrayList<Tensor>, Tensor) -> Unit)
}