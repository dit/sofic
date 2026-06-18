/* Add a [>>>] button on the top-right corner of code samples to hide
 * the >>> and ... prompts and the output and thus make the code
 * copyable.
 */

$(document).ready(function() {

    var div = $('.highlight-python .highlight,' +
                '.highlight-py .highlight,' +
                '.highlight-sage .highlight,' +
                '.highlight-python3 .highlight,' +
                '.highlight-py3 .highlight,' +
                '.highlight-pycon .highlight,' +
                '.highlight-pycon3 .highlight,' +
                '.highlight-pytb .highlight,' +
                '.highlight-py3tb .highlight,' +
                '.highlight-ipython .highlight,' +
                '.highlight-ipython3 .highlight,' +
                '.highlight-ipythontb .highlight,' +
                '.highlight-ipythontb3 .highlight,' +
                '.highlight-ipythoncon .highlight,' +
                '.highlight-ipython3con .highlight,' +
                '.highlight-ipy .highlight,' +
                '.highlight-ipy3 .highlight')


    var pre = div.find('pre');

    pre.parent().parent().css('position', 'relative');
    var hide_text = 'Hide the prompts and output';
    var show_text = 'Show the prompts and output';
    var border_width = pre.css('border-top-width');
    var border_style = pre.css('border-top-style');
    var border_color = pre.css('border-top-color');
    var button_styles = {
        'cursor':'pointer', 'position': 'absolute', 'top': '0', 'right': '0',
        'border-color': border_color, 'border-style': border_style,
        'border-width': border_width, 'color': border_color,
        'padding-left': '0.3em',  'padding-right': '0.3em',
        'border-radius': '0 3px 0 0'
    }

    div.each(function(index) {
        var jthis = $(this);
        if (jthis.find('.gp').length > 0) {
            var button = $('<i class="copybutton fa fa-eye fa-1g"></i>');
            button.css(button_styles)
            button.attr('title', hide_text);
            jthis.prepend(button);
        }
        jthis.find('pre:has(.gt)').contents()
            .filter(function() {
                return ((this.nodeType == 3) && (this.data.trim().length > 0));
            })
                .wrap('<span>')
            .end();
    });

    $('.copybutton').toggle(
        function() {
            var button = $(this);
            button.parent().find('.go, .gp, .gh, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.go, .gp, .gh').css('visibility', 'hidden');
            button.attr('class', 'copybutton fa fa-eye-slash fa-1g');
            button.attr('title', show_text);
        },
        function() {
            var button = $(this);
            button.parent().find('.go, .gp, .gh, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go, .gh').css('visibility', 'visible');
            button.attr('class', 'copybutton fa fa-eye fa-1g');
            button.attr('title', hide_text);
        });
});
