// This function returns the cookies in a manageable state

import { clearCookies } from "./clearCookies";

// To be used in Python.js to set inital values for FormInput
export function readCookies() {
    var pairs = document.cookie.split(";");
    var cookies = {};

    if (pairs.length > 1) {
      // if cookies exist, read them
      var i = 0
      for (; i<pairs.length-1; i++){
        var pair = pairs[i].split("=");
        cookies[(pair[0]+'').trim()] = pair.slice(1).join('=');
      }

      // handle the checked array seperately
      var checked = pairs[i].split("=");
      var checkedArray = checked[1].split(",");
      cookies[(checked[0]+'').trim()] = checkedArray;
    } else {
      // if no cookies are set, return these default values
      cookies = {
        checked: [ "INNE", "LOF", "SIL", "COPOD", "IF" ],
        maxConstraintPercent: "0.000003",
        numberOfClusters: "3",
        questionsPerIteration: "20",
        reduction_algorithm: "PCA"
      }
    }
    
    return cookies;
}