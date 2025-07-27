// For correct display of Pandas Styled tables
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('table[id^="T_"]').forEach(table => {
        table.classList.add('dataframe');
    });
});