document.addEventListener('DOMContentLoaded', function () {
  const checkButton = document.getElementById('checkButton');
  const addToBlacklistButton = document.getElementById('addToBlacklist');
  const showBlacklistButton = document.getElementById('showBlacklist');
  const removeBlacklistButton = document.getElementById('removeBlacklist');

  const urlInput = document.getElementById('urlInput');
  const feature1Checkbox = document.getElementById('feature1');
  const feature2Checkbox = document.getElementById('feature2');
 

  // Initialize checkboxes from localStorage
  feature1Checkbox.checked = localStorage.getItem('feature1') === 'true';
  feature2Checkbox.checked = localStorage.getItem('feature2') === 'true';


  feature1Checkbox.addEventListener('change', function() {
    localStorage.setItem('feature1', feature1Checkbox.checked);
  });

  feature2Checkbox.addEventListener('change', function() {
    localStorage.setItem('feature2', feature2Checkbox.checked);
  });


  checkButton.addEventListener('click', function() {
    const inputValue = urlInput.value.trim();
    if (inputValue !== '') {
      checkPhishing(inputValue);
    }
  });

  //  removeBlacklistButton.addEventListener('click', function() {
  //    localStorage.removeItem('blacklist');
  //   alert('Blacklist cleared successfully.');
  // });

   removeBlacklistButton.addEventListener('click', function() {
    chrome.storage.local.remove('blacklist', function() {
      alert('Blacklist cleared successfully.');
    });
  });

  addToBlacklistButton.addEventListener('click', function() {
    const url = urlInput.value.trim();
    if (url) {
      addToBlacklist(url);
    }
  });

  showBlacklistButton.addEventListener('click', showBlacklist);


  function displayResult(data, checkType) {
  const resultDiv = document.getElementById('result');
  let resultHTML = '';

  if (data.isPhishing) {
    resultHTML = `<p>Phishing Detected for ${checkType} with ${data.confidenceLevel}% confidence.</p>`;
    resultDiv.style.color = data.confidenceLevel >= 75 ? 'red' :
                          data.confidenceLevel >= 50 ? 'orange' : 'green';
  } else {
    resultHTML = `<p>No Phishing Detected for ${checkType}.</p>`;
    resultDiv.style.color = 'green';
  }

  // Optionally, adjust the style based on the confidence level or result
  // resultDiv.style.color = data.confidenceLevel >= 75 ? 'red' :
  //                         data.confidenceLevel >= 50 ? 'orange' : 'green';

  resultDiv.innerHTML = resultHTML;
  }

  function handleError(error) {
  console.error('Error:', error);
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '<p>An error occurred while checking for phishing.</p>';
  resultDiv.style.color = 'red'; // Indicate error visually
}
  function checkPhishing(inputValue) {
    let promises = [];
    let checkedFeatures = [];

    if (feature1Checkbox.checked) {
      promises.push(fetchPhishingData('http://127.0.0.1:5000/check_phishing_catboost', { url: inputValue }));
      checkedFeatures.push('CatBoost');
    }
    
    if (feature2Checkbox.checked) {
      promises.push(fetchPhishingData('http://127.0.0.1:5000/check_phishing_bert', { url: inputValue }));
      checkedFeatures.push('BERT');
    }

    Promise.all(promises).then(results => {
      if (results.length > 1) {
        // Average confidence if multiple checks are done
        const averageConfidence = results.reduce((acc, result) => acc + result.confidenceLevel, 0) / results.length;
        const isPhishing = results.some(result => result.isPhishing);
        displayResult({ isPhishing, confidenceLevel: averageConfidence }, checkedFeatures.join(' and '));
      } else if (results.length === 1) {
        // Display result for single check
        displayResult(results[0], checkedFeatures[0]);
      }
    }).catch(error => {
      console.error('Error:', error);
    });
}

  function fetchPhishingData(apiUrl, requestData) {
    return fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    })
    .then(response => response.json())
    .catch(error => {
      console.error('Fetch error:', error);
      throw error;
    });
  }

  // function addToBlacklist(url) {
  //   let blacklist = JSON.parse(localStorage.getItem('blacklist')) || [];
  //   if (!blacklist.includes(url)) {
  //     blacklist.unshift(url); // Add new URL at the beginning
  //     localStorage.setItem('blacklist', JSON.stringify(blacklist));
  //     alert('URL added to blacklist successfully.');
  //   } else {
  //     alert('URL is already in the blacklist.');
  //   }
  // }
  function addToBlacklist(url) {
    chrome.storage.local.get(['blacklist'], function(result) {
      let blacklist = result.blacklist || [];
      if (!blacklist.includes(url)) {
        blacklist.unshift(url); // Add new URL at the beginning
        chrome.storage.local.set({ 'blacklist': blacklist }, function() {
          alert('URL added to blacklist successfully.');
        });
      } else {
        alert('URL is already in the blacklist.');
      }
    });
  }

  // function showBlacklist() {
  //   let blacklist = JSON.parse(localStorage.getItem('blacklist')) || [];
  //   if (blacklist.length > 0) {
  //     let blacklistString = blacklist.join('<br>');
  //     createAndShowModal('Blacklisted URLs:', blacklistString);
  //   } else {
  //     alert('The blacklist is empty.');
  //   }
  // }

  function showBlacklist() {
    chrome.storage.local.get(['blacklist'], function(result) {
      let blacklist = result.blacklist || [];
      if (blacklist.length > 0) {
        let blacklistString = blacklist.join('<br>');
        createAndShowModal('Blacklisted URLs:', blacklistString);
      } else {
        alert('The blacklist is empty.');
      }
    });
  }

  function createAndShowModal(title, content) {
    let modal = document.getElementById('blacklistModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.setAttribute('id', 'blacklistModal');
      modal.classList.add('modal');

      const modalContent = document.createElement('div');
      modalContent.classList.add('modal-content');

      const closeBtn = document.createElement('span');
      closeBtn.classList.add('close');
      closeBtn.innerHTML = '&times;';
      closeBtn.onclick = function() { modal.style.display = 'none'; };

      const titleElem = document.createElement('h2');
      titleElem.textContent = title;

      modalContent.appendChild(closeBtn);
      modalContent.appendChild(titleElem);
      modal.appendChild(modalContent);
      document.body.appendChild(modal);

      // Click outside to close
      window.onclick = function(event) {
        if (event.target == modal) {
          modal.style.display = 'none';
        }
      };
    }

    const contentElem = document.createElement('p');
    contentElem.innerHTML = content;
    modal.querySelector('.modal-content').appendChild(contentElem);

    modal.style.display = 'block';
  }
});
