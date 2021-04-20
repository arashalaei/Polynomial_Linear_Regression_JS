/* jslint esversion: 9 */

/**
 * @author Arash Alaei <arashalaei22@gmail.com>
 * @since 02/00/2021
 */

require('@tensorflow/tfjs-node');
const dfd = require('danfojs-node'); // import pandas as pd
const PolynomialRegression  = require('ml-regression').PolynomialRegression;
const tf = require('@tensorflow/tfjs'); // import numpy as np
const path = require("path");

(async() => {
    // Importing the dataset
    const dataset = await dfd.read_csv(`file://${path.join(__dirname,'./Position_Salaries.csv')}`);

    let x = tf.util.flatten(dataset.iloc({rows:[':'], columns:['1']}).values);
    let y = tf.util.flatten(dataset.iloc({rows:[':'], columns:['2']}).values);
    
    let regressor = new PolynomialRegression(x, y, 4);
    console.log(regressor.predict(6.5));
})();