package layers

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.function.Executable
import structure.Shape
import structure.Tensor

internal class MaxPooling2DTest {

    companion object {
        private var inTensor = arrayListOf<Tensor>()
        @BeforeAll
        @JvmStatic
        fun setup() {
            this.inTensor = arrayListOf(Tensor(Shape(16, 1), false), Tensor(Shape(24, 1), false))
            inTensor.first().elements = floatArrayOf(
                    1F, 2F, 3F, 4F,
                    5F, 6F, 7F, 8F,
                    9F, 10F, 11F, 12F,
                    13F, 14F, 15F, 16F,
                    1F, 2F, 3F, 4F,
                    5F, 6F, 7F, 9F)
            inTensor.last().elements = floatArrayOf(
                    20F, 34F, 21F, 27F,
                    50F, 35F, 23F, 36F,
                    25F, 28F, 30F, 34F,
                    32F, 26F, 31F, 22F,
                    1F, 2F, 3F, 4F,
                    5F, 6F, 7F, 9F)
        }

    }

    @Test
    fun getOutputShape() {
    }

    @Test
    fun updateWeights() {
    }

    @Test
    fun forward() {
        val max = MaxPooling2D(Shape(4, 6), Shape(2, 2), 2)
        val outTensor = arrayListOf(Tensor(Shape(4, 1), false), Tensor(Shape(6, 1), false))

        max.forward(inTensor, outTensor)

        val actualFirst = floatArrayOf(
                6F, 8F,
                14F, 16F,
                6F, 9F)
        val actualSecond = floatArrayOf(
                50F, 36F,
                32F, 34F,
                6F, 9F)
        val actualFirstMask = intArrayOf(
                -1, -1, -1, -1,
                -1, 0, -1, 1,
                -1, -1, -1, -1,
                -1, 2 - 1, 3)
        val actualSecondMask = intArrayOf(
                -1, 0 - 1, 1
                - 1, -1, -1, -1,
                -1, -1, -1, -1,
                -1, 2, 3, -1)


        Assertions.assertAll(
                Executable { Assertions.assertArrayEquals(actualFirst, outTensor[0].elements) },
                Executable { Assertions.assertArrayEquals(actualSecond, outTensor[1].elements) }
                //Executable { Assertions.assertArrayEquals(actualFirstMask, max.marks[0]) },
                //Executable { Assertions.assertArrayEquals(actualSecondMask, max.marks[1]) }
        )
    }

    @Test
    fun backward() {
    }
}