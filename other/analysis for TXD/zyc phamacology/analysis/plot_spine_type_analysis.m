b=bar(reshape(m,[2,4])','BarWidth',1);
b(1).FaceColor='g';b(2).FaceColor='w';
xticklabels({'mushroom','stubby','thin','all'});
set(gcf,'Color','w');
box('off');
ylabel('spine density(# per 100\mum)');

groupw=2/(2+1.5);
barx=reshape(reshape([(1:4)-groupw/4 (1:4)+groupw/4],[4,2])',[1,8]);
hold on;eb=errorbar(barx,m,sd);hold off;
eb.LineStyle='none';eb.Color='black';

ster_x=barx(1:2:end);
ster_y=m(1:2:end)+sd(1:2:end);
for n=1:3
    text(ster_x(n),ster_y(n),'***',...
        'HorizontalAlignment','center',...
        'VerticalAlignment','bottom');
end
text(ster_x(4),ster_y(4),'n.s.',...
    'HorizontalAlignment','center',...
    'VerticalAlignment','bottom');

lg=legend('ro25','saline','Location','northwest');
legend('boxoff');