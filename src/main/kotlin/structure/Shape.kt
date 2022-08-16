package structure

class Shape(vararg axisList: Int) {
    var axis = axisList
    val volume = {
        var product = 1
        for (i in axisList) {
            product *= i
        }
        product
    }()

    fun transpose(): Shape {
        return Shape(axis[1], axis[0])
    }
}