class RPSDataset {
  constructor() {
    this.labels = []
  }
  // You can add an example to this dataset using the below function
  addExample(example, label) {
    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.labels.push(label);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }
  
  // This function will encode the ground truth labels
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
