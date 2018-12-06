// https://stackoverflow.com/questions/23939588/how-to-animate-drawing-lines-on-canvas

(function() { var lastTime = 0; var vendors = ['webkit', 'moz']; for(var 
    x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
        window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
        window.cancelAnimationFrame =
          window[vendors[x]+'CancelAnimationFrame'] || window[vendors[x]+'CancelRequestAnimationFrame'];
    }

    if (!window.requestAnimationFrame)
        window.requestAnimationFrame = function(callback, element) {
            var currTime = new Date().getTime();
            var timeToCall = Math.max(0, 16 - (currTime - lastTime));
            var id = window.setTimeout(function() { callback(currTime + timeToCall); },              timeToCall);
            lastTime = currTime + timeToCall;
            return id;
        };

    if (!window.cancelAnimationFrame)
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
}());


var ticks = 0;
animLoop();

function animLoop() {
  window.requestAnimationFrame(animLoop);
  animhelpDrawframe(ticks);
  ticks += 1;
}

//var canvas=document.getElementById("myCanvas");
//var ctx=canvas.getContext("2d");
/*
function animhelp_framedraw(ticks)
{
		// a simple clock animation
		ctx.clearRect(0, 0, 400, 400);
        if (ticks==0) { ctx.moveTo(0,0, 0, 0); };
		ctx.beginPath();
		ctx.moveTo(200,200);
        ctx.lineTo(200+Math.cos(ticks/6)*200,
					200+Math.sin(ticks/6)*200);
        ctx.stroke();
}
*/

