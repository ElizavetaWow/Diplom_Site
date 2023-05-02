let multiselect_block = document.querySelectorAll(".multiselect_block");
multiselect_block.forEach(parent => {
    let label = parent.querySelector(".field_multiselect");
    let select = parent.querySelector(".custom_field_select ");
    let text = label.innerHTML;
    if (select.selectedOptions.length) {
        label.innerHTML = "";
        for (let option of select.selectedOptions) {
            let button = document.createElement("button");
            button.type = "button";
            button.className = "btn_multiselect";
            button.textContent = option.label;
            label.append(button);
        }
    }

    select.addEventListener("change", function (element) {
        let selectedOptions = this.selectedOptions;
        label.innerHTML = "";
        for (let option of selectedOptions) {
            let button = document.createElement("button");
            button.type = "button";
            button.className = "btn_multiselect";
            button.textContent = option.label;
            label.append(button);
        }

    })

})
$('body').click(function (event) {
    if (!$(event.target).closest('.custom_multiselect').length && !$(event.target).is('.custom_multiselect')) {
        $(".multiselect_checkbox").prop('checked', false)
    }
});


window.onmousedown = function (e) {
    var el = e.target;
    if (el.tagName.toLowerCase() == 'option' && el.parentNode.hasAttribute('multiple')) {
        e.preventDefault();
        var select = el.parentNode;
        var scroll = select.scrollTop;

        if (el.hasAttribute('selected')) el.removeAttribute('selected');
        else el.setAttribute('selected', '');

        el.parentNode.dispatchEvent(new Event("change"))
        setTimeout(function () { select.scrollTop = scroll; }, 0);


    }
}