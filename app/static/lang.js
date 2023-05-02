
var dataTableLanguages = {
    'es': '//cdn.datatables.net/plug-ins/1.10.21/i18n/Spanish.json',
    'en': '//cdn.datatables.net/plug-ins/1.13.4/i18n/en-GB.json',
    'ru': '//cdn.datatables.net/plug-ins/1.13.4/i18n/ru.json',
};

var language = dataTableLanguages[navigator.language];
var lang_opts = {};

if (language) {
    if (typeof language === 'string') {
        lang_opts = {"sUrl": language};
    } else {
        lang_opts = language;
    }
}