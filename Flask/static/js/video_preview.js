const inputFile_d = document.getElementById("dance_upload");
const video_d = document.getElementById("preview1");

const inputFile_ud = document.getElementById("user_dance_upload");
const video_ud = document.getElementById("preview2");

const inputFile_u360 = document.getElementById("user_360_upload");
const video_u360 = document.getElementById("preview3");



inputFile_d.addEventListener("change", function() {
    const file = document.getElementById("dance_upload").files[0];
    const videourl = URL.createObjectURL(file);
    document.getElementById("preview1").setAttribute("src", videourl);
    document.getElementById("preview1").play();
})

inputFile_ud.addEventListener("change", function() {
    const file = document.getElementById("user_dance_upload").files[0];
    const videourl = URL.createObjectURL(file);
    document.getElementById("preview2").setAttribute("src", videourl);
    document.getElementById("preview2").play();
})

inputFile_u360.addEventListener("change", function() {
    const file = document.getElementById("user_360_upload").files[0];
    const videourl = URL.createObjectURL(file);
    document.getElementById("preview3").setAttribute("src", videourl);
    document.getElementById("preview3").play();
})