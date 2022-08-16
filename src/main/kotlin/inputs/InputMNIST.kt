package inputs

import structure.Shape
import structure.Tensor
import java.io.File

class InputMNIST(path_train: String) : Input {
    override val input: List<Tensor>
    override val labels: List<Tensor>

    init {
        val bufferedReader = File(path_train).bufferedReader()
        val lineList = mutableListOf<String>()

        bufferedReader.useLines { lines -> lines.forEach { lineList.add(it) } }

        input = mutableListOf()
        labels = mutableListOf()

        lineList.forEach {
            val values = it.split(",")

            val label = Tensor(Shape(10, 1), false)
            val pixels = Tensor(Shape(784, 1), false)

            label.elements[values[0].toInt()] = 1F
            values.subList(1, values.size).forEachIndexed { i, pixel -> pixels.elements[i] = (pixel.toFloat() / 255F) }

            input.add(pixels)
            labels.add(label)
        }
    }
}