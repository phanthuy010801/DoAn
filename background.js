function guid() {
    function s4() {
        return Math.floor((1 + Math.random()) * 0x10000)
            .toString(16)
            .substring(1);
    }
    return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
}

chrome.storage.local.get({idnEnable: true}, function(items) {
    const idnEnable = items.idnEnable;
    if (items.userGuid == null) {
        var userGuid = guid();
        console.log(userGuid);
        chrome.storage.local.set({
            userGuid: userGuid
        }, function() {
            addListener(idnEnable);
        });
    } else {
        addListener(idnEnable);
    }
});

var isDomainIDN = function(domain) {
    return (domain.startsWith('https') || domain.startsWith('http') || domain.startsWith('www.'));
};

function checkPhishing(url, callback) {
    const apiUrl = 'http://127.0.0.1:5000/check_phishing_bert';

    const requestData = {
        url: url
    };

    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
    })
    .then(response => response.json())
    .then(data => {
        const {
            isPhishing,
            confidenceLevel
        } = data;
        callback(isPhishing, confidenceLevel); // Call the callback function with the confidenceLevel
    })
    .catch(error => {
        console.log('Error:', error);
    });
}

var extractHostname = function(url) {
    var hostname;
    //find & remove protocol (http, ftp, etc.) and get hostname

    if (url.indexOf("://") > -1) {
        hostname = url.split('/')[2];
    }
    else {
        hostname = url.split('/')[0];
    }

    //find & remove port number
    hostname = hostname.split(':')[0];
    //find & remove "?"
    hostname = hostname.split('?')[0];

    return hostname;
};

function addToBlacklist(url) {
    // Retrieve the existing blacklist from chrome.storage.local
    chrome.storage.local.get(['blacklist'], function(result) {
        let blacklist = result.blacklist || [];

        if (!blacklist.includes(url)) {
            blacklist.push(url);
            // Save the updated blacklist back to chrome.storage.local
            chrome.storage.local.set({'blacklist': blacklist}, function() {
                console.log('Blacklist updated.');
            });
        }
    });
}


function delURLfromBL(url){
    console.log("Entering delURLfromBL function with URL:", url);  // Kiểm tra khi hàm được gọi

    chrome.storage.local.get(['blacklist'], function(result) {
        let blacklist = result.blacklist || [];

        if (blacklist.includes(url)) {
            console.log("URL found in blacklist:", url);  // Kiểm tra xem URL có trong blacklist không
            blacklist = blacklist.filter(u => u !== url); // Xóa URL khỏi blacklist
            
            // Save the updated blacklist back to chrome.storage.local
            chrome.storage.local.set({'blacklist': blacklist}, function() {
                console.log('URL removed from blacklist:', url);  // Xác nhận URL đã bị xóa
            });
        } else {
            console.log("URL not found in blacklist:", url);  // Kiểm tra nếu URL không nằm trong blacklist
        }
    });
}
/*
if (result.originalUrl) {
    // Remove the URL from the blacklist if it's present
    let blacklist = result.blacklist || [];
    if (blacklist.includes(result.originalUrl)) {
        blacklist = blacklist.filter(url => url !== result.originalUrl);
        chrome.storage.local.set({ blacklist: blacklist }, function() {
            console.log('URL removed from blacklist.');
        });
    }
    
    // Redirect the tab to the original URL
    chrome.tabs.update(sender.tab.id, { url: result.originalUrl });
}*/

function addToWhitelist(url) {
    // Retrieve the existing whitelist from chrome.storage.local
    chrome.storage.local.get(['whitelist'], function(result) {
        let whitelist = result.whitelist || [];

        if (!whitelist.includes(url)) {
            whitelist.push(url);
            // Save the updated whitelist back to chrome.storage.local
            chrome.storage.local.set({'whitelist': whitelist}, function() {
                console.log('Whitelist updated.');
            });
        }
    });
}

function addListener(idnEnable) {
    chrome.tabs.onUpdated.addListener(function mylistener(tabId, changedProps, tab) {
        if (changedProps.status != "complete") {
            return;
        }

        console.log(tab.url);

        if (idnEnable) {
            // Kiểm tra xem URL có nằm trong blacklist không
            chrome.storage.local.get(['whitelist'], function(result) {
                let whitelist = result.whitelist || [];

                if (!whitelist.includes(tab.url)) {
                    checkPhishing(tab.url, function(isPhishing, confidenceLevel) {
                        if (isPhishing > 0 && confidenceLevel > 90 && isDomainIDN(tab.url)) {
                            addToBlacklist(tab.url);
                            chrome.storage.local.set({ originalUrl: tab.url }, function() {
                                chrome.tabs.update(tabId, { url: "page_blocked.html" });
                            });
                            return;
                        }
                    });
                }
            });
        }
    });
}

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "closeTab") {
        // addToBlacklist(tab.url);
        chrome.tabs.remove(sender.tab.id);
    } else if (request.action === "continueToSite") {
        // Retrieve the original URL from storage or context
        chrome.storage.local.get(['originalUrl'], function(result) {
            if (result.originalUrl) {
                // delURLfromBL(result.originalUrl);
                addToWhitelist(result.originalUrl)
                chrome.tabs.update(sender.tab.id, { url: result.originalUrl });
            }
            
        });

    }
});
