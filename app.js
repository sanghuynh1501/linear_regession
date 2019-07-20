const csv = require('csv-parser')
const fs = require('fs')
const accounting = require('accounting')

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}  

function read_csv(file_link) {
    let x_vectors = []
    let y_vectors = []
    let max_square = 0
    let max_year = 0
    let results = []
    let streets = ['Pave', 'Grvl', undefined]
    let neighborhoods = [
        'CollgCr', 'Veenker', 'Crawfor',
        'NoRidge', 'Mitchel', 'Somerst',
        'NWAmes',  'OldTown', 'BrkSide',
        'Sawyer',  'NridgHt', 'NAmes',
        'SawyerW', 'IDOTRR',  'MeadowV',
        'Edwards', 'Timber',  'Gilbert',
        'StoneBr', 'ClearCr', 'NPkVill',
        'Blmngtn', 'BrDale',  'SWISU',
        'Blueste'
    ]
    return new Promise((resolve) => {
        fs.createReadStream(file_link)
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('end', () => {
                results.forEach(item => {
                    if (item.square > max_square) {
                        max_square = item.square
                    }
                    if (item.salePrice > max_year) {
                        max_year = item.salePrice
                    }
                    let x_vector = [
                        // parseFloat(item.square) / 9986,
                        // parseFloat(streets.indexOf(item.street)),
                        // parseFloat(neighborhoods.indexOf(item.neighborhood)),
                        parseFloat(item.yearBuilt) / 2009,
                        // parseFloat(item.yearSold) / 2010,
                    ]
                    let y_vector = [
                        parseFloat(item.salePrice / 99900)
                    ]
                    x_vectors.push(x_vector)
                    y_vectors.push(y_vector)
                })
                resolve({
                    x: x_vectors,
                    y: y_vectors
                });
            })
    })
}

async function read_data(data_path) {
    let train = await read_csv(data_path + '/train.csv')
    let test = await read_csv(data_path + '/test.csv')
    return {
        train,
        test
    }
}

function multiply(a, b) {
    var aNumRows = a.length, aNumCols = a[0].length,
        bNumRows = b.length, bNumCols = b[0].length,
        m = new Array(aNumRows)  // initialize array of rows
    for (var r = 0; r < aNumRows; ++r) {
      m[r] = new Array(bNumCols) // initialize the current row
      for (var c = 0; c < bNumCols; ++c) {
        m[r][c] = 0;             // initialize the current cell
        for (var i = 0; i < aNumCols; ++i) {
          m[r][c] += a[r][i] * b[i][c]
        }
      }
    }
    return m
}

function add(a, b) {
    var res = []
    a.forEach((t, n1) => {
        let sums = []
        t.forEach((num, n2) => {
        sums.push(num + b[n1][n2])
        });
        res.push(sums)
    })
    return res
}

function minus(a, b) {
    var res = []
    a.forEach((t, n1) => {
        let sums = []
        t.forEach((num, n2) => {
        sums.push(num - b[n1][n2])
        });
        res.push(sums)
    })
    return res
}

function square(x) {
    return multiply(x, x)
}

function random_matrix(r, c) {
    let matrix = []
    for (let i = 0; i < r; i++) {
        let row = []
        for (let j = 0; j < c; j++) {
            row.push(Math.random())
        }
        matrix.push(row)
    }
    return matrix
}

function linear_regession(x, w, b) {
    // x * w + b
    let dupplicate_b = []
    for (i = 0; i < x.length; i++) {
        dupplicate_b.push(b[0])
    }
    return add(multiply(x, w), dupplicate_b)
}

function loss_function(x, y, w, b) {
    let loss = 0
    for (let i = 0; i < x.length; i++) {
        loss += Math.abs(minus([y[i]], linear_regession([x[i]], w, b))[0][0])
    }
    return loss / (x.length * 2)
}

function dot(number, x) {
    for (let i = 0; i < x.length; i++) {
        for (let j = 0; j < x[i].length; j++) {
            x[i][j] = x[i][j] * number
        }
    }
    return x
}

function trainport(x) {
     // Calculate the width and height of the Array
     var a = x,
     w = a.length ? a.length : 0,
     h = a[0] instanceof Array ? a[0].length : 0;

    // In case it is a zero matrix, no transpose routine needed.
    if (h === 0 || w === 0) {
        return [];
    }

    var i, j, t = [];

    // Loop through every item in the outer array (height)
    for (i = 0; i < h; i++) {

        // Insert a new row (array)
        t[i] = [];

        // Loop through every item per item in outer array (width)
        for (j = 0; j < w; j++) {

            // Save transposed data.
            t[i][j] = a[j][i];
        }
    }
    return t
}

function grad_weight (x, y, w, b) {
    let N = x.length
    return dot(1/N, multiply(trainport(x), minus(multiply(x, w), y)))
}

function grad_bias (x, y, w, b) {
    let N = x.length
    return dot(1/N, linear_regession(x, y, w, b))
}

function gradient_descent_weight(weight, train, weight, bias, learning_rate) {
    return minus(weight, dot(learning_rate, grad_weight (train.x, train.y, weight, bias)))
}

function gradient_descent_bias(weight, train, weight, bias, learning_rate) {
    return minus(bias, dot(learning_rate, grad_bias (train.x, train.y, weight, bias)))
}

async function main() {
    let { train, test } = await read_data('./data')
    let weight = random_matrix(1, 1)
    let bias = random_matrix(1, 1)
    let learning_rate = 0.002
    let dem = 0
    loss = loss_function(train.x, train.y, weight, bias)
    console.log('loss ', loss)
    while (dem < 10000) {
        weight = gradient_descent_weight(weight, train, weight, bias, learning_rate)
        bias = gradient_descent_bias(weight, train, weight, bias, learning_rate)
        loss = loss_function(train.x, train.y, weight, bias)
        console.log(loss)
        dem++
    }

    // for(let i = 0; i < test.x.length; i++) {
    //     let y_pred = linear_regession([test.x[i]], weight, bias)
    //     console.log('y: ' + accounting.format(test.y[i] * 99900) + ', y_pred: ' + accounting.format(y_pred * 99900))
    // }
}

main()