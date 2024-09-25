chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "htmlRetrieved") {
    var retrievedHtml = request.data;
    console.log("Retrieved HTML:", retrievedHtml);
    // You can perform further actions with the retrieved HTML here
  }
});
