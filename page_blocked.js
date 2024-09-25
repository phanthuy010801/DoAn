document.getElementById('btn_safety').addEventListener('click', function() {
    // chrome.runtime.sendMessage({action: "closeTab"});
    window.close()
});

document.getElementById('btn_continue').addEventListener('click', function() {
    chrome.runtime.sendMessage({action: "continueToSite"});
});
