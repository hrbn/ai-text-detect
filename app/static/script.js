async function modelPredict(form) {
  const textInput = form.textinput.value;
  const resultDiv = document.getElementById('result');

  // Clear previous result
  resultDiv.innerHTML = '';

  if (textInput.trim() === '') {
    resultDiv.innerHTML = 'Please enter some text!';
    return;
  }

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: textInput })
    });

    const data = await response.json();

    if (data.prediction) {
      resultDiv.innerHTML = `<section class="level">
  <div class="level-item has-text-centered">
    <div>
      <p class="heading">Prediction</p>
      <p class="title">${data.prediction}</p>
    </div>
  </div>
  <div class="level-item has-text-centered">
    <div>
      <p class="heading">Probability</p>
      <p class="title">${(data.probability * 100).toFixed(2)}%</p>
    </div>
  </div>
  </section>`

    } else {
      resultDiv.innerHTML = `<article class="message is-danger">
  <div class="message-body">
    <strong>Error:</strong> ${data.error}
  </div>
</article>`
    }
  } catch (error) {
      resultDiv.innerHTML = `<article class="message is-danger">
  <div class="message-body">
    <strong>Error:</strong> ${error.message}
  </div>
</article>`
  }
}