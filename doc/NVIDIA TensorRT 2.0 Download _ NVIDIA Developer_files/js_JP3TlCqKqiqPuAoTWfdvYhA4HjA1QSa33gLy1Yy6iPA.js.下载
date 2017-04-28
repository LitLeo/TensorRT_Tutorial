/**
 * @file
 * Contains UX enchancements for codefilter.module.
 */

(function ($) {

  Drupal.behaviors.codefilter = {
    attach: function (context) {
      var $expandablePre = $('pre.codeblock.nowrap-expand', context);
      // Stop if the expandable pre is null.
      // For non prism pages or if the feature is turned off.
      if (!$expandablePre[0]) {
        return;
      }
      // Getting padding as we can't get CSS attribute selectors through JS.
      var em = Number($expandablePre.css('font-size').replace(/[^\d]/g, ''));

      // Provide expanding text boxes when code blocks are too long.
      $expandablePre.find('code').each(function () {
        var $code = $(this);
        var $pre = $code.parent();
        var contents_width = $code.width() + (em * 2);
        var width = $pre.width() + (em * 2);

        if (contents_width > width) {
          $pre.hover(function () {
            $pre.css('width', width).animate({ width: contents_width + 'px' }, {
              duration: 100,
              queue: false
            });
          },
          function () {
            $pre.css('width', contents_width).animate({ width: width + 'px' }, {
              duration: 100,
              queue: false
            });
          });
        }
      });
    }
  }

})(jQuery);
;
(function ($) {
  Drupal.behaviors.gssAutocomplete = {
    attach: function(context, settings) {
      if (settings.gss.key == undefined) {
        return;
      }

      $('.block-search .form-item-search-block-form input.form-text, .gss .form-item-keys input.form-text, .block-search .form-search input.form-text')
        .focus(function () {
          this.select();
        })
        .mouseup(function (e) {
          e.preventDefault();
        })
        .autocomplete({
          position: {
            my: "left top",
            at: "left bottom",
            offset: "0, 5",
            collision: "none"
          },
          source: function (request, response) {
            $.ajax({
              url: "http://clients1.google.com/complete/search?q=" + request.term + "&hl=en&client=partner&source=gcsc&partnerid=" + settings.gss.key + "&ds=cse&nocache=" + Math.random().toString(),
              dataType: "jsonp",
              success: function (data) {
                response($.map(data[1], function (item) {
                  return {
                    label: item[0],
                    value: item[0]
                  };
                }));
              }
            });
          },
          autoFill: true,
          minChars: 0,
          select: function (event, ui) {
            $(this).closest('input').val(ui.item.value);
            $(this).closest('form').trigger('submit');
          }
        });
    }
  };
}(jQuery));
;
